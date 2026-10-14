"""Paired physical comparison of the frozen video curve and a straight transfer.

The new model changes only intermediate transfer XY readout rows. Both variants
use the V5 camera, learned vision/text features, servo and restricted policy API.
True object state is logged and scored after actions; it never schedules them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import torch

from .integrity import environment_signature, sha256_file
from .motion_ablation_model import changed_readout_rows, fit_straight_model
from .sim import CONTROL_DT
from .vla_control import VLAController
from .vla_env import VLAEnv, score_target
from .vla_evaluate import (_group_summary, normalized_source_hash, run_physical_case,
                          validate_checkpoint_metadata, validate_receipt)
from .vla_policy import PRIMARY_INSTRUCTIONS, load_vla_policy, save_vla_policy
from .vla_train import _validate_arrays


SCENE_SEED = 20261027
VARIANTS = ('video_motion', 'straight_transfer')
DISTANCE_BANDS_M = ((.045, .075), (.075, .110), (.110, .160))
EXPECTED_CASES = 96
EXPECTED_ATTEMPTS = 192
MAX_STEPS = 800
AIRBORNE_Z_M = .125
DEFAULT_V5_RECEIPT = 'outputs/vla_v5/model_freeze.json'
DEFAULT_PROTOCOL = 'docs/superpowers/specs/2026-10-14-motion-ablation-design.md'
NEW_MODULES = ('motion_ablation', 'motion_ablation_model', 'motion_ablation_analysis')
FINAL_READOUT_NAMES = ('coefficient_head.2.weight', 'coefficient_head.2.bias')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_new_json(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def file_binding(path, *, source=False):
    return dict(path=Path(path).as_posix(), checksum='normalized_source' if source else 'bytes',
        sha256=normalized_source_hash(path) if source else sha256_file(path))


def check_bindings(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Missing experiment file bindings')
    for label, binding in bindings.items():
        mode = binding.get('checksum')
        if mode not in ('normalized_source', 'bytes'):
            raise ValueError(f'Unknown checksum mode for {label}')
        digest = (normalized_source_hash(binding['path']) if mode == 'normalized_source'
                  else sha256_file(binding['path']))
        if digest != binding.get('sha256'):
            raise ValueError(f'Frozen {label} bytes/source differ')
    return True


def runtime_contract():
    return dict(python_version=platform.python_version(), torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda, numpy_version=np.__version__, mujoco_version=mujoco.__version__,
        inference_device='cpu', torch_num_threads=2)


def _require_runtime():
    if torch.__version__ != '2.14.0+cpu' or torch.version.cuda is not None:
        raise ValueError('The declared paired experiment uses CPU PyTorch 2.14.0+cpu')
    torch.set_num_threads(2)


def scene_configurations(*, seed=SCENE_SEED):
    if isinstance(seed, bool) or seed != SCENE_SEED:
        raise ValueError('The predeclared motion-ablation seed is 20261027')
    rng = np.random.default_rng(seed)
    cases = []
    for band, bounds in enumerate(DISTANCE_BANDS_M):
        for direction in range(8):
            angle_lower = -np.pi+direction*np.pi/4
            for placement in range(4):
                distance = float(rng.uniform(*bounds))
                angle = float(rng.uniform(angle_lower, angle_lower+np.pi/4))
                delta = distance*np.array([np.cos(angle), np.sin(angle)])
                low = np.maximum([.40, -.10], np.array([.32, -.23])+delta)
                high = np.minimum([.52, .10], np.array([.62, .23])+delta)
                if np.any(low > high):
                    raise ValueError('Declared workspace has no feasible bearing interval')
                bearing = rng.uniform(low, high)
                cases.append(dict(configuration_id=f'M6_d{band}_a{direction}_p{placement}',
                    distance_band_index=band, direction_bin_index=direction, position_index=placement,
                    distance_band_m=list(bounds), endpoint_distance_m=distance, angle_rad=angle,
                    ring_xy_m=(bearing-delta).tolist(), bearing_xy_m=bearing.tolist(),
                    bearing_minus_ring_xy_m=delta.tolist(), reset_seed=int(rng.integers(0, 2**31-1)),
                    ancestry='procedural geometry; no human trajectory ancestor'))
    return cases


def evaluation_schedule(cases):
    if cases != scene_configurations():
        raise ValueError('Paired schedule requires all 96 predeclared fresh scenes')
    return [dict(case, variant=variant, suite=variant, instruction_index=0,
            instruction=PRIMARY_INSTRUCTIONS['inside'], ablation='none',
            attempt_id=f'{case["configuration_id"]}:{variant}')
        for case in cases for variant in VARIANTS]


def _load_original_context(receipt_path):
    receipt = _read_json(receipt_path)
    manifest = _read_json(receipt['manifest_path'])
    validate_receipt(receipt, manifest)
    model, metadata = load_vla_policy(receipt['checkpoint_path'])
    validate_checkpoint_metadata(metadata, manifest)
    return receipt, manifest, model, metadata


def load_train_arrays(dataset_manifest):
    """Load TRAIN only; other split files are hash-bound without np.load."""
    path = Path(dataset_manifest)
    manifest = _read_json(path)
    train_path = path.parent/'train.npz'
    if sha256_file(train_path) != manifest['files']['train.npz']['sha256']:
        raise ValueError('Training array hash differs from the inherited dataset')
    with np.load(train_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    _validate_arrays(arrays, 'train', manifest)
    return arrays


def shared_parameter_audit(base, straight):
    """Verify the intervention mask against actual checkpoint tensors."""
    before, after = base.state_dict(), straight.state_dict()
    if before.keys() != after.keys():
        raise ValueError('Model state keys differ between variants')
    changed = changed_readout_rows()
    common = np.setdiff1d(np.arange(23*4*10), changed)
    def equal_bytes(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().cpu().contiguous().view(torch.uint8),
                            right.detach().cpu().contiguous().view(torch.uint8)))
    shared = all(equal_bytes(before[name], after[name]) for name in before if name not in FINAL_READOUT_NAMES)
    rows = all(equal_bytes(before[name][common], after[name][common]) for name in FINAL_READOUT_NAMES)
    if not shared or not rows:
        raise ValueError('A shared visual/text/critical-action parameter changed')
    return dict(unchanged_parameters_equal=shared, common_readout_rows_equal=rows,
        changed_row_indices=changed.tolist(), shared_readout_row_count=len(common))


def prepare(output, *, v5_receipt=DEFAULT_V5_RECEIPT, protocol_path=DEFAULT_PROTOCOL):
    """Fit and seal both variants, scenes and sources before physical testing."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new experiment directory; sealed outputs are immutable')
    _require_runtime()
    inherited_receipt, inherited_manifest, base, metadata = _load_original_context(v5_receipt)
    data_path = inherited_manifest['dataset_manifest_path']
    arrays = load_train_arrays(data_path)
    straight, fit = fit_straight_model(base, arrays)
    shared = shared_parameter_audit(base, straight)
    if (not fit['base_model_unchanged'] or not fit['common_predicted_outputs_equal']
            or fit['held_out_data_used']):
        raise ValueError('Straight fitting did not preserve the declared controlled intervention')
    sources = {name: f'ring_learning/{name}.py' for name in NEW_MODULES}
    fit.update(base_checkpoint_sha256=inherited_receipt['checkpoint_sha256'],
        dataset_manifest_sha256=inherited_manifest['dataset_manifest_sha256'],
        training_npz_sha256=inherited_manifest['dataset_files']['train.npz']['sha256'],
        implementation_sources={name: normalized_source_hash(path) for name, path in sources.items()},
        runtime_contract=runtime_contract())
    output.mkdir(parents=True, exist_ok=False)
    fit_path, straight_path = output/'fit_report.json', output/'straight_policy.pt'
    _write_new_json(fit_path, fit)
    inherited_keys = ('dataset_manifest_sha256', 'train_npz_sha256', 'val_npz_sha256',
                      'training_sources', 'validation_sources', 'implementation_sources')
    straight_metadata = {name: metadata[name] for name in inherited_keys}
    straight_metadata.update(inherited_v5_metadata=metadata,
        ablation_fit='TRAIN-only final linear readout; no new epoch or model selection',
        action_label_origin='Intermediate XY straight chords from TRAIN endpoints; all other V5 actions preserved',
        motion_ablation=dict(
        base_checkpoint_sha256=inherited_receipt['checkpoint_sha256'], fit_report_sha256=sha256_file(fit_path),
        intervention='Only waypoints4..18 XY final-readout rows refitted on TRAIN; recorded endpoints retained',
        implementation_sources=fit['implementation_sources']))
    save_vla_policy(straight_path, straight, straight_metadata)
    bindings = dict(video_motion=file_binding(inherited_receipt['checkpoint_path']),
        straight_transfer=file_binding(straight_path), fit_report=file_binding(fit_path),
        inherited_v5_receipt=file_binding(v5_receipt),
        inherited_v5_manifest=file_binding(inherited_receipt['manifest_path']),
        inherited_v5_training_report=file_binding(inherited_receipt['training_report_path']),
        dataset_manifest=file_binding(data_path),
        train_npz=file_binding(inherited_manifest['dataset_files']['train.npz']['path']),
        val_npz=file_binding(inherited_manifest['dataset_files']['val.npz']['path']),
        protocol=file_binding(protocol_path))
    for name, item in inherited_manifest['implementation_sources'].items():
        bindings[f'source_{name}'] = file_binding(item['path'], source=True)
    for name, path in sources.items():
        bindings[f'source_{name}'] = file_binding(path, source=True)
    for path in sorted(Path('vendor/panda').rglob('*')):
        if path.is_file():
            bindings[f'physics_asset_{path.as_posix()}'] = file_binding(path)
    cases = scene_configurations()
    manifest = dict(schema_version=1, protocol='video_motion_partial_ablation_v6', seed=SCENE_SEED,
        configurations=cases, schedule=evaluation_schedule(cases), expected_cases=EXPECTED_CASES,
        expected_attempts=EXPECTED_ATTEMPTS, variants=list(VARIANTS), bindings=bindings,
        models={variant: bindings[variant] for variant in VARIANTS}, shared_weights=shared,
        runtime_contract=runtime_contract(), environment=environment_signature(),
        execution=dict(max_steps=MAX_STEPS, transfer_pre_waypoint_indices=list(range(4, 20)),
            ring_airborne_z_m=AIRBORNE_Z_M, score_used_for_termination=False,
            success_definition='controller finished and post-execution inside target score valid',
            true_object_state='Metrics, reset and scoring only; never a runtime action input'),
        intervention='TRAIN-only intermediate transfer XY readout; personally recorded endpoint geometry retained',
        outcome_filtering=False, denominator_policy='All 192 registered attempts; failures, exceptions and missing attempts retained')
    validate_manifest(manifest)
    manifest_path = output/'manifest.json'
    _write_new_json(manifest_path, manifest)
    receipt = dict(schema_version=1, protocol=manifest['protocol'],
        manifest_path=manifest_path.as_posix(), manifest_sha256=sha256_file(manifest_path),
        frozen_utc=datetime.now(timezone.utc).isoformat(), frozen_before_test_execution=True,
        variants=list(VARIANTS), models=manifest['models'], bindings=bindings,
        runtime_contract=manifest['runtime_contract'])
    _write_new_json(output/'freeze_receipt.json', receipt)
    return manifest


def validate_manifest(manifest):
    if (manifest.get('schema_version') != 1 or manifest.get('protocol') != 'video_motion_partial_ablation_v6'
            or manifest.get('seed') != SCENE_SEED or manifest.get('expected_cases') != EXPECTED_CASES
            or manifest.get('expected_attempts') != EXPECTED_ATTEMPTS or manifest.get('variants') != list(VARIANTS)
            or manifest.get('outcome_filtering') is not False):
        raise ValueError('Unsupported paired motion-ablation protocol')
    if manifest.get('configurations') != scene_configurations():
        raise ValueError('Frozen scene geometry differs from the predeclared draws')
    if manifest.get('schedule') != evaluation_schedule(manifest['configurations']):
        raise ValueError('Frozen paired attempt schedule differs')
    check_bindings(manifest['bindings'])
    if (manifest.get('environment') != environment_signature()
            or manifest.get('runtime_contract') != runtime_contract()):
        raise ValueError('Frozen physics/runtime differs from the declared paired experiment')
    for variant in VARIANTS:
        if manifest.get('models', {}).get(variant) != manifest['bindings'].get(variant):
            raise ValueError('Checkpoint binding differs from the registered model')
    execution = manifest.get('execution', {})
    if (execution.get('max_steps') != MAX_STEPS or execution.get('ring_airborne_z_m') != AIRBORNE_Z_M
            or execution.get('transfer_pre_waypoint_indices') != list(range(4, 20))
            or execution.get('score_used_for_termination') is not False):
        raise ValueError('Frozen execution/scoring protocol differs')
    inherited = _read_json(manifest['bindings']['inherited_v5_receipt']['path'])
    validate_receipt(inherited, _read_json(inherited['manifest_path']))
    return True


def validate_freeze_receipt(receipt, manifest_path):
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    if (receipt.get('schema_version') != 1 or receipt.get('protocol') != manifest['protocol']
            or receipt.get('frozen_before_test_execution') is not True
            or receipt.get('manifest_sha256') != sha256_file(manifest_path)
            or receipt.get('bindings') != manifest['bindings']
            or receipt.get('models') != manifest['models']
            or receipt.get('runtime_contract') != manifest['runtime_contract']):
        raise ValueError('Frozen manifest/model receipt differs')
    return True


class TracedController:
    """Record the original controller's own stage without changing its action."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.controller_waypoint_indices = []

    def __call__(self, env):
        self.controller_waypoint_indices.append(int(self.delegate.waypoint_index))
        return self.delegate(env)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


def path_metrics(points):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Path samples must be finite N x 2 XY coordinates')
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) else None
    chord = float(np.linalg.norm(points[-1]-points[0])) if len(points) else None
    degenerate = chord is None or chord <= 1e-9
    lateral = None if degenerate else (points-points[0])@np.array([
        -(points[-1]-points[0])[1], (points[-1]-points[0])[0]])/chord
    return dict(samples=len(points), path_length_m=length, chord_length_m=chord,
        arc_to_chord_ratio=None if degenerate else length/chord,
        max_abs_lateral_m=None if lateral is None else float(np.abs(lateral).max()),
        mean_abs_lateral_m=None if lateral is None else float(np.abs(lateral).mean()), degenerate=degenerate)


def transfer_metrics(row):
    trajectory = row.get('trajectory', [])
    stages = row.get('controller_waypoint_indices', [])[:max(0, len(trajectory)-1)]
    indices = [i for i, stage in enumerate(stages) if 4 <= stage <= 19]
    start, end = (indices[0], indices[-1]+1) if indices else (None, None)
    selected = trajectory[start:end+1] if indices else []
    result = dict(transfer_steps=len(indices), transfer_duration_s=len(indices)*CONTROL_DT,
        transfer_start_trajectory_index=start, transfer_end_trajectory_index=end,
        ring_airborne_z_m=AIRBORNE_Z_M,
        ring_airborne_fraction=float(np.mean([s['ring_xyz_m'][2] > AIRBORNE_Z_M for s in selected])) if selected else None,
        ring_contact_fraction=float(np.mean([s['finger_contacts'] > 0 for s in selected])) if selected else None)
    for name in ('ring', 'tool', 'command'):
        result[name] = path_metrics([s[f'{name}_xyz_m'][:2] for s in selected])
    predicted = row.get('predicted_waypoints')
    predicted = np.asarray(predicted, float) if predicted is not None else np.empty((0, 4))
    result['predicted'] = path_metrics(predicted[3:20, :2] if predicted.shape == (23, 4) else [])
    result['terminal_center_error_m'] = (float(np.linalg.norm(
        np.asarray(trajectory[-1]['ring_xyz_m'])[:2]-np.asarray(trajectory[-1]['bearing_xyz_m'])[:2]))
        if trajectory else None)
    return result


def run_case(env, request, controller, *, scorer=score_target):
    traced = TracedController(controller)
    row = run_physical_case(env, request, traced, scorer=scorer)
    row['controller_waypoint_indices'] = traced.controller_waypoint_indices
    row['transfer_metrics'] = transfer_metrics(row)
    return row


def summarize(rows, schedule):
    if schedule != evaluation_schedule(scene_configurations()):
        raise ValueError('Summary requires all 192 predeclared registered attempts')
    registered = {r['attempt_id']: r for r in schedule}
    if len(registered) != len(schedule):
        raise ValueError('Duplicate registered attempt')
    seen = set()
    for row in rows:
        identifier = row.get('attempt_id')
        if identifier not in registered or identifier in seen:
            raise ValueError('Duplicate or unregistered evaluation attempt')
        if any(row.get(key) != value for key, value in registered[identifier].items()):
            raise ValueError('Attempt identity differs from the fixed paired schedule')
        seen.add(identifier)
    paired = dict(complete_pairs=0, same_initial_rgb=0, common_predicted_outputs_equal=0,
        rgb_pairs_compared=0, prediction_pairs_compared=0, initial_rgb_mismatches=0,
        common_prediction_mismatches=0,
        changed_transfer_predictions=0, maximum_common_prediction_difference=0.,
        same_final_ring_xyz=0, maximum_final_ring_xy_difference_m=0., same_target_scores=0,
        same_success_flags=0, both_successes=0, video_only_successes=0, straight_only_successes=0)
    lookup = {(r['configuration_id'], r['variant']): r for r in rows}
    common = np.ones((23, 4), dtype=bool)
    common[4:19, :2] = False
    for identifier in sorted({r['configuration_id'] for r in schedule}):
        first, second = (lookup.get((identifier, variant)) for variant in VARIANTS)
        if first is None or second is None:
            continue
        paired['complete_pairs'] += 1
        if first.get('input_image_sha256') and second.get('input_image_sha256'):
            paired['rgb_pairs_compared'] += 1
            equal = first['input_image_sha256'] == second['input_image_sha256']
            paired['same_initial_rgb'] += equal
            paired['initial_rgb_mismatches'] += not equal
        paired['same_target_scores'] += first.get('target_score') == second.get('target_score')
        paired['same_success_flags'] += first.get('success') == second.get('success')
        paired['both_successes'] += bool(first.get('success') and second.get('success'))
        paired['video_only_successes'] += bool(first.get('success') and not second.get('success'))
        paired['straight_only_successes'] += bool(second.get('success') and not first.get('success'))
        if first.get('predicted_waypoints') is not None and second.get('predicted_waypoints') is not None:
            a, b = np.asarray(first['predicted_waypoints'], float), np.asarray(second['predicted_waypoints'], float)
            if a.shape == b.shape == (23, 4):
                paired['prediction_pairs_compared'] += 1
                equal = a[common].tobytes() == b[common].tobytes()
                paired['common_predicted_outputs_equal'] += equal
                paired['common_prediction_mismatches'] += not equal
                paired['maximum_common_prediction_difference'] = max(paired['maximum_common_prediction_difference'],
                    float(np.abs(a[common]-b[common]).max()))
                paired['changed_transfer_predictions'] += bool(np.linalg.norm(a[4:19, :2]-b[4:19, :2], axis=1).max() > 1e-9)
        if first.get('trajectory') and second.get('trajectory'):
            a, b = (np.asarray(r['trajectory'][-1]['ring_xyz_m'], float) for r in (first, second))
            paired['same_final_ring_xyz'] += a.tobytes() == b.tobytes()
            paired['maximum_final_ring_xy_difference_m'] = max(paired['maximum_final_ring_xy_difference_m'],
                float(np.linalg.norm(a[:2]-b[:2])))
    return dict(by_variant={v: _group_summary(rows, [r for r in schedule if r['variant'] == v]) for v in VARIANTS},
        by_distance_band={v: {str(b): _group_summary(rows, [r for r in schedule if r['variant'] == v and r['distance_band_index'] == b])
            for b in range(3)} for v in VARIANTS},
        by_direction_bin={v: {str(b): _group_summary(rows, [r for r in schedule if r['variant'] == v and r['direction_bin_index'] == b])
            for b in range(8)} for v in VARIANTS}, paired=paired,
        policy_forbidden_access_attempts=sum(r.get('policy_forbidden_access_attempts', 0) for r in rows),
        attempted_rollouts=len(rows), expected_rollouts=len(schedule),
        denominator_policy='All registered attempts; missing, failed and exceptional cases retained')


def run(manifest_path, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Evaluation output exists; choose a new reproduction destination')
    _require_runtime()
    manifest = _read_json(manifest_path)
    receipt_path = Path(manifest_path).parent/'freeze_receipt.json'
    receipt = _read_json(receipt_path)
    validate_freeze_receipt(receipt, manifest_path)
    models = {v: load_vla_policy(manifest['models'][v]['path'])[0] for v in VARIANTS}
    shared = shared_parameter_audit(models['video_motion'], models['straight_transfer'])
    if shared != manifest['shared_weights']:
        raise ValueError('Loaded checkpoints differ from the frozen shared-weight audit')
    env, rows = VLAEnv(max_steps=MAX_STEPS), []
    try:
        for index, request in enumerate(manifest['schedule']):
            controller = VLAController(models[request['variant']], request['instruction'])
            rows.append(run_case(env, request, controller))
            if (index+1) % 12 == 0:
                print(f'Motion ablation {index+1}/{EXPECTED_ATTEMPTS}', flush=True)
    finally:
        env.close()
    summary = summarize(rows, manifest['schedule'])
    paired = summary['paired']
    fair = (paired['same_initial_rgb'] == paired['common_predicted_outputs_equal'] == EXPECTED_CASES)
    report = dict(schema_version=1, protocol=manifest['protocol'],
        manifest_sha256=sha256_file(manifest_path), freeze_receipt_sha256=sha256_file(receipt_path),
        model_sha256={v: manifest['models'][v]['sha256'] for v in VARIANTS},
        runtime=dict(runtime_contract(), python_executable=sys.executable,
                     actual_torch_num_threads=torch.get_num_threads()),
        summary=summary, rollouts=rows, paired_control_checks_passed=fair,
        object_oracle_used_for_actions=False, test_outcomes_used_for_model_selection=False,
        transfer_metric_definition='Actual samples in pre-controller-waypoint stages4..19; lateral departure from each path own endpoint chord',
        ring_airborne_definition='Fraction of logged transfer samples with actual ring center Z >0.125m; no outcome-based filtering')
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(destination, report)
    if not fair:
        raise RuntimeError(f'Paired RGB/critical outputs were not identical; all attempted rows retained in {destination}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--out', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--manifest', required=True)
    run_parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.out)
        print(json.dumps(dict(cases=EXPECTED_CASES, attempts=EXPECTED_ATTEMPTS,
                              model_sha256={v: manifest['models'][v]['sha256'] for v in VARIANTS})))
    else:
        report = run(args.manifest, args.out)
        print(json.dumps(report['summary']['by_variant']))


if __name__ == '__main__':
    main()
