"""Frozen-scene evaluation of the explicitly hybrid video motion controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import make_configurations, split_episodes, validate_lineage, write_json
from .integrity import environment_signature, sha256_file


def prepare(real_path, destination, seed=20261017):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('V3 scenes are frozen; use a new destination')
    real = json.loads(Path(real_path).read_text(encoding='utf-8'))
    splits = split_episodes(real['episodes'], 20261014)
    configs = make_configurations(splits, dict(train=1, val=2, test=6, ood_test=6),
                                  seed=seed, angle_limit=np.pi, translation_limit=.015)
    rng = np.random.default_rng(seed + 101)
    for split in ('val', 'test', 'ood_test'):
        for cfg in configs[split]:
            old = np.asarray(cfg['bearing_xy_m'])
            delta = np.asarray(cfg['ring_xy_m']) - old
            bearing = rng.uniform([.40, -.10], [.52, .10])
            cfg['bearing_xy_m'] = bearing.tolist()
            cfg['ring_xy_m'] = (bearing + delta).tolist()
            cfg['augmentation']['bearing_relocation_m'] = (bearing-old).tolist()
    validate_lineage(configs)
    manifest = dict(protocol='video_motion_v3_frozen_new_scene_draws', seed=seed,
                    source_split_seed=20261014, environment=environment_signature(),
                    real_episodes_sha256=sha256_file(real_path),
                    source_video_sha256=real['source']['sha256'], configurations=configs,
                    frozen_before_V3_training=True,
                    limitation='Same original held-out sources as V1/V2; correlated synthetic descendants, '
                               'not new human recordings or an independent real-world benchmark.',
                    history='V1/V2 outcomes informed the task and workflow; these scene draws are new.')
    write_json(destination, manifest)
    return manifest


def summarize(rows):
    summaries = {}
    for split, controller in sorted({(r['split'], r['controller']) for r in rows}):
        group = [r for r in rows if r['split'] == split and r['controller'] == controller]
        failures = {}
        for row in group:
            if not row['success']:
                reason = row['info'].get('failure_reason') or 'unknown'
                failures[reason] = failures.get(reason, 0) + 1
        nsuccess = sum(r['success'] for r in group)
        summaries[f'{split}/{controller}'] = dict(
            successes=nsuccess, episodes=len(group), success_rate=nsuccess/len(group),
            source_count=len({r['source_episode_id'] for r in group}), failure_counts=failures,
            learned_steps=sum(r['learned_steps'] for r in group),
            planned_steps=sum(r['planned_steps'] for r in group))
    return summaries


def validate_model_provenance(metadata, manifest, splits):
    if metadata.get('real_episodes_sha256') != manifest['real_episodes_sha256']:
        raise ValueError('Checkpoint endpoint manifest differs from frozen scenes')
    train_ids = set(metadata['training_sources'])
    configs = manifest['configurations']
    expected_train = {c['source_episode_id'] for c in configs['train']}
    if not train_ids or not train_ids.issubset(expected_train):
        raise ValueError('Motion model training sources do not belong to frozen training split')
    for split in splits:
        if split != 'train' and train_ids & {c['source_episode_id'] for c in configs[split]}:
            raise ValueError('Motion model source leakage into evaluation')


def evaluate(manifest_path, checkpoint, output, splits=('test', 'ood_test'), limit=None):
    from .evaluate import validate_evaluation_request
    from .sim import RingEnv
    from .video_policy import VideoController, load_video_policy

    manifest_path, output = Path(manifest_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['environment'] != environment_signature():
        raise ValueError('Frozen scenes belong to a different simulation version')
    configs = manifest['configurations']
    validate_evaluation_request(configs, splits, limit)
    validate_lineage(configs)
    model, metadata = load_video_policy(checkpoint)
    validate_model_provenance(metadata, manifest, splits)
    env = RingEnv()
    rows = []
    try:
        for split in splits:
            cases = configs[split] if limit is None else configs[split][:limit]
            for name in ('learned_motion', 'straight_path'):
                for i, cfg in enumerate(cases):
                    obs = env.reset(cfg['ring_xy_m'], cfg['bearing_xy_m'], cfg['reset_seed'])
                    controller = VideoController(model, geometry=name == 'straight_path')
                    trajectory, references, info = [], [], {}
                    forward_calls = 0
                    for step in range(env.max_steps):
                        old_phase = env.phase
                        action = np.clip(controller(env, obs), -1., 1.)
                        if old_phase == 4:
                            if (name == 'learned_motion' and 0 < controller.progress < 1
                                    and np.linalg.norm(controller.goal_xy-controller.start_xy) >= 1e-12):
                                forward_calls += 1
                            trajectory.append(env.ring_xyz[:2].tolist())
                            # Generated reference is saved separately from physical ring motion.
                            references.append(env.command_xyz[:2].tolist())
                        obs, done, info = env.step(action)
                        if done:
                            break
                    row = dict(controller=name, split=split,
                               configuration_id=cfg['configuration_id'],
                               source_episode_id=cfg['source_episode_id'], success=env.success(),
                               steps=step+1, info=info, learned_steps=controller.learned_steps,
                               planned_steps=controller.planned_steps,
                               neural_forward_calls=forward_calls,
                               transfer_ring_xy_m=trajectory, transfer_command_xy_m=references)
                    rows.append(row)
                    if (i+1) % 5 == 0 or i+1 == len(cases):
                        score = sum(r['success'] for r in rows if r['split'] == split and r['controller'] == name)
                        print(f'{split}/{name} {i+1}/{len(cases)} successful={score}', flush=True)
                        write_json(output/'rollouts.json', rows)
    finally:
        env.close()
        write_json(output/'rollouts.json', rows)
    report = dict(summaries=summarize(rows), checkpoint_sha256=sha256_file(checkpoint),
                  manifest_sha256=sha256_file(manifest_path), training_metadata=metadata,
                  protocol=manifest['protocol'],
                  controller_scope='Learned XY transfer; geometric approach, grip, vertical lift, '
                                   'lower, release and retract. Shared FSM, IK, servo and physics.',
                  evaluation_uses_human_test_motion=False,
                  step_count_definition='learned_steps means learned-transfer scope, including deterministic '
                                        'endpoint steps; neural_forward_calls counts actual forward passes. '
                                        'Z and gripper remain geometric during transfer. planned_steps '
                                        'counts other stages and all zero-residual baseline stages.',
                  placement_criteria='Full ring projection in hole with 0.3mm margin, tilt <=8deg, '
                                     'table height within1.5mm, speed<0.01m/s, angular speed<0.15rad/s, '
                                     'open fingers, zero finger contact, settled >=0.4s.',
                  limitation=manifest['limitation'])
    write_json(output/'results.json', report)
    print(json.dumps(report['summaries']), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--seed', type=int, default=20261017, help='Scene draw seed for --prepare')
    parser.add_argument('--real', default='data/real_episodes.json')
    parser.add_argument('--manifest', default='outputs/video_v3/configurations.json')
    parser.add_argument('--checkpoint', default='outputs/video_model/policy.pt')
    parser.add_argument('--output', default='outputs/video_v3/evaluation')
    parser.add_argument('--split', nargs='+', default=['test', 'ood_test'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.real, args.manifest, args.seed)
    else:
        evaluate(args.manifest, args.checkpoint, args.output, args.split, args.limit)


if __name__ == '__main__':
    main()
