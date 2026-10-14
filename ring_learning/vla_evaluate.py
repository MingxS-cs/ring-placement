"""Frozen V5 RGB/text/action evaluation with a restricted runtime policy API.

Procedural scenes have no human trajectory ancestry. True object poses and
contact information are collected only outside the policy for grading and
reports. The controller receives a restricted view containing RGB and its own
proprioception, so forbidden oracle reads fail visibly rather than silently
changing the action path. No evaluation result participates in model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .integrity import environment_signature, sha256_file


SCENE_SEED = 20261024
DISTANCE_BANDS_M = ((.045, .075), (.075, .110), (.110, .160))
ANGLE_BINS = 8
PLACEMENTS_PER_CELL = 4
EXPECTED_CASES = 96
MAX_STEPS = 800
COMMANDS = ('put the ring inside the bearing', 'put the ring left of the bearing',
            'put the ring right of the bearing')
SUITES = ('main', 'black_image', 'language_counterfactual')
DEFAULT_IMPLEMENTATIONS = {name: f'ring_learning/{name}.py' for name in
    ('vla_policy', 'vla_control', 'vla_env', 'vla_data', 'vla_train', 'vla_evaluate')}
DEFAULT_PROTOCOL = 'docs/superpowers/specs/2026-10-14-tiny-vla-design.md'


def normalized_source_hash(path):
    text = Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _write_new_json(destination, value):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def scene_configurations(*, seed=SCENE_SEED):
    """Draw the complete fixed strata before observing any physical outcome."""
    if isinstance(seed, bool) or seed != SCENE_SEED:
        raise ValueError('The predeclared V5 scene seed is 20261024')
    rng = np.random.default_rng(seed)
    cases = []
    for band, bounds in enumerate(DISTANCE_BANDS_M):
        for direction in range(ANGLE_BINS):
            angle_lower = -np.pi+direction*2*np.pi/ANGLE_BINS
            for placement in range(PLACEMENTS_PER_CELL):
                distance = float(rng.uniform(*bounds))
                angle = float(rng.uniform(angle_lower, angle_lower+2*np.pi/ANGLE_BINS))
                delta = distance*np.array([np.cos(angle), np.sin(angle)])
                lower = np.maximum([.40, -.10], np.array([.32, -.23])+delta)
                upper = np.minimum([.52, .10], np.array([.62, .23])+delta)
                if np.any(lower > upper):
                    raise ValueError('Declared reset workspace has no feasible bearing interval')
                bearing = rng.uniform(lower, upper)
                ring = bearing-delta
                cases.append(dict(configuration_id=f'G5_d{band}_a{direction}_p{placement}',
                    distance_band_index=band, direction_bin_index=direction, position_index=placement,
                    distance_band_m=list(bounds), endpoint_distance_m=distance, angle_rad=angle,
                    ring_xy_m=ring.tolist(), bearing_xy_m=bearing.tolist(),
                    bearing_minus_ring_xy_m=delta.tolist(),
                    conditional_bearing_interval_m=dict(lower=lower.tolist(), upper=upper.tolist()),
                    reset_seed=int(rng.integers(0, 2**31-1)),
                    ancestry='independent procedural geometry; no human source trajectory ancestor'))
    return cases


def evaluation_schedule(cases):
    if cases != scene_configurations():
        raise ValueError('Evaluation scenes differ from the complete fixed V5 geometry')
    requests = []
    for suite, ablation, chosen, instructions in (
            ('main', 'none', range(EXPECTED_CASES), (0,)),
            ('black_image', 'black_image', range(EXPECTED_CASES), (0,)),
            ('language_counterfactual', 'none', range(0, EXPECTED_CASES, 4), (0, 1, 2))):
        for index in chosen:
            for instruction in instructions:
                case = cases[index]
                requests.append(dict(case, suite=suite, ablation=ablation,
                    instruction_index=instruction, instruction=COMMANDS[instruction],
                    repeat_of_main=bool(suite == 'language_counterfactual' and instruction == 0),
                    attempt_id=f'{suite}:{case["configuration_id"]}:i{instruction}'))
    return requests


def prepare(dataset_manifest, destination, *, protocol_path=DEFAULT_PROTOCOL,
            implementation_paths=None):
    """Bind fixed geometry to final development data and implementation sources."""
    if Path(destination).exists():
        raise FileExistsError('V5 geometry manifest is immutable')
    paths = DEFAULT_IMPLEMENTATIONS if implementation_paths is None else implementation_paths
    if not isinstance(paths, dict) or not paths:
        raise ValueError('Implementation source paths must be a nonempty mapping')
    data = _read_json(dataset_manifest)
    files = data.get('files', {})
    dataset_files = {}
    for name in ('train.npz', 'val.npz'):
        digest = files.get(name, {}).get('sha256')
        path = Path(dataset_manifest).parent/name
        if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
            raise ValueError('Development dataset array file hash differs or is missing')
        dataset_files[name] = dict(path=path.as_posix(), sha256=digest)
    manifest = dict(schema_version=1, protocol='vla_v5_fixed_geometry_96', seed=SCENE_SEED,
        configurations=scene_configurations(), expected_cases=EXPECTED_CASES,
        dataset_manifest_path=Path(dataset_manifest).as_posix(),
        dataset_manifest_sha256=sha256_file(dataset_manifest),
        dataset_files=dataset_files, source_splits=data.get('split_sources'),
        source_video_sha256=data.get('source_video_sha256', data.get('source', {}).get('sha256')),
        protocol_path=Path(protocol_path).as_posix(), protocol_sha256=sha256_file(protocol_path),
        implementation_sources={name: dict(path=Path(path).as_posix(), sha256=normalized_source_hash(path))
            for name, path in sorted(paths.items())}, environment=environment_signature(),
        geometry_protocol=dict(distance_bands_m=[list(bounds) for bounds in DISTANCE_BANDS_M],
            direction_bins=ANGLE_BINS, placements_per_cell=PLACEMENTS_PER_CELL,
            ring_bounds_m=dict(lower=[.32, -.23], upper=[.62, .23]),
            bearing_bounds_m=dict(lower=[.40, -.10], upper=[.52, .10]),
            sampling='uniform distance/angle in each cell and conditional feasible bearing interval',
            max_steps=MAX_STEPS, source_trajectory_ancestry=False),
        evaluation_protocol=dict(suites=list(SUITES), main_cases=96, black_image_cases=96,
            language_scene_indices=list(range(0, EXPECTED_CASES, 4)), language_instructions=list(COMMANDS),
            language_attempts=72, language_inside_repeats_of_main=24,
            total_attempts=264, frozen_model_selection='development validation only; never evaluation outcomes',
            policy_api='RGB and own proprioception only; forbidden attributes blocked at runtime',
            object_coordinates='reset, record and score only; never passed to runtime policy',
            score_used_for_termination=False, outcome_filtering=False,
            success_definition='controller finished and post-execution target score valid'),
        denominator_policy='All registered attempts including exceptions, timeouts and unexecuted cases',
        scope='Same object dimensions, fixed robot-hidden simulated RGB camera; new positions/directions/distances',
        outcome_filtering=False)
    validate_manifest(manifest)
    _write_new_json(destination, manifest)
    return manifest


def validate_manifest(manifest):
    if (manifest.get('schema_version') != 1 or manifest.get('protocol') != 'vla_v5_fixed_geometry_96'
            or manifest.get('seed') != SCENE_SEED or manifest.get('expected_cases') != EXPECTED_CASES
            or manifest.get('outcome_filtering') is not False):
        raise ValueError('Unsupported frozen V5 geometry protocol')
    if manifest.get('configurations') != scene_configurations():
        raise ValueError('Frozen geometry scenes differ from predeclared draws')
    if manifest.get('environment') != environment_signature():
        raise ValueError('Frozen physics environment signature differs')
    if sha256_file(manifest['dataset_manifest_path']) != manifest.get('dataset_manifest_sha256'):
        raise ValueError('Frozen development dataset manifest differs')
    for name in ('train.npz', 'val.npz'):
        binding = manifest.get('dataset_files', {}).get(name, {})
        if not binding or sha256_file(binding['path']) != binding.get('sha256'):
            raise ValueError('Frozen development dataset arrays differ')
    if sha256_file(manifest['protocol_path']) != manifest.get('protocol_sha256'):
        raise ValueError('Frozen selection/evaluation protocol differs')
    sources = manifest.get('implementation_sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Missing implementation source bindings')
    for binding in sources.values():
        if normalized_source_hash(binding['path']) != binding.get('sha256'):
            raise ValueError('Frozen implementation source differs')
    protocol = manifest.get('evaluation_protocol', {})
    if (protocol.get('total_attempts') != 264 or protocol.get('main_cases') != 96
            or protocol.get('black_image_cases') != 96 or protocol.get('language_attempts') != 72
            or protocol.get('language_scene_indices') != list(range(0, EXPECTED_CASES, 4))
            or protocol.get('language_instructions') != list(COMMANDS)
            or protocol.get('score_used_for_termination') is not False
            or protocol.get('success_definition') != 'controller finished and post-execution target score valid'
            or protocol.get('outcome_filtering') is not False):
        raise ValueError('Frozen evaluation schedule or grading boundary differs')
    return True


def validate_checkpoint_metadata(metadata, manifest):
    """Require the selected model to use the same arrays and source episodes."""
    for key, expected in (
            ('dataset_manifest_sha256', manifest['dataset_manifest_sha256']),
            ('train_npz_sha256', manifest['dataset_files']['train.npz']['sha256']),
            ('val_npz_sha256', manifest['dataset_files']['val.npz']['sha256'])):
        if metadata.get(key) != expected:
            raise ValueError('Selected checkpoint training dataset binding differs')
    splits = manifest.get('source_splits', {})
    train = metadata.get('training_sources')
    val = metadata.get('validation_sources')
    if (not isinstance(train, list) or not isinstance(val, list)
            or sorted(train) != sorted(splits.get('train', []))
            or sorted(val) != sorted(splits.get('val', []))
            or set(train).intersection(val)):
        raise ValueError('Selected checkpoint source membership differs or overlaps')
    model_sources = metadata.get('implementation_sources', {})
    for name in ('vla_policy', 'vla_train'):
        if name not in manifest['implementation_sources']:
            continue
        digest = model_sources.get(name)
        if isinstance(digest, dict):
            digest = digest.get('sha256')
        if digest != manifest['implementation_sources'][name]['sha256']:
            raise ValueError('Selected checkpoint implementation source differs')
    return True


def freeze_model_receipt(manifest_path, checkpoint, training_report, destination):
    """Freeze the already selected weights; never select or tune a checkpoint."""
    if Path(destination).exists():
        raise FileExistsError('V5 model/evaluation receipt is immutable')
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    training = _read_json(training_report)
    selection = training.get('metadata', {}).get('selection', {})
    selected_epoch = selection.get('selected_epoch', training.get('selected_epoch'))
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch < 0:
        raise ValueError('Training report does not identify a valid selected epoch')
    receipt = dict(schema_version=1, protocol=manifest['protocol'],
        frozen_utc=datetime.now(timezone.utc).isoformat(), frozen_before_test_execution=True,
        manifest_path=Path(manifest_path).as_posix(), manifest_sha256=sha256_file(manifest_path),
        checkpoint_path=Path(checkpoint).as_posix(), checkpoint_sha256=sha256_file(checkpoint),
        training_report_path=Path(training_report).as_posix(), training_report_sha256=sha256_file(training_report),
        dataset_manifest_sha256=manifest['dataset_manifest_sha256'], protocol_sha256=manifest['protocol_sha256'],
        selected_epoch=selected_epoch, selection_record=selection,
        implementation_sources=manifest['implementation_sources'])
    validate_receipt(receipt, manifest)
    _write_new_json(destination, receipt)
    return receipt


def validate_receipt(receipt, manifest):
    validate_manifest(manifest)
    if (receipt.get('schema_version') != 1 or receipt.get('protocol') != manifest['protocol']
            or receipt.get('frozen_before_test_execution') is not True):
        raise ValueError('Unsupported model freeze receipt')
    for label, path_key, digest_key in (
            ('manifest', 'manifest_path', 'manifest_sha256'),
            ('checkpoint', 'checkpoint_path', 'checkpoint_sha256'),
            ('training report', 'training_report_path', 'training_report_sha256')):
        if sha256_file(receipt[path_key]) != receipt.get(digest_key):
            raise ValueError(f'Frozen {label} bytes differ')
    training = _read_json(receipt['training_report_path'])
    selection = training.get('metadata', {}).get('selection', {})
    if (receipt.get('selected_epoch') != selection.get('selected_epoch', training.get('selected_epoch'))
            or receipt.get('selection_record') != selection):
        raise ValueError('Frozen checkpoint selection record differs')
    if (receipt.get('dataset_manifest_sha256') != manifest['dataset_manifest_sha256']
            or receipt.get('protocol_sha256') != manifest['protocol_sha256']
            or receipt.get('implementation_sources') != manifest['implementation_sources']):
        raise ValueError('Frozen data, protocol or implementation binding differs')
    return True


class ControllerEnvironmentView:
    """Allow only runtime RGB and own-state APIs; count rejected oracle accesses."""
    allowed_attributes = frozenset(('observe', 'render_policy', 'tool_xyz', 'command_xyz',
                                    'finger_opening', 'steps'))

    def __init__(self, env):
        object.__setattr__(self, '_environment', env)
        object.__setattr__(self, 'forbidden_access_attempts', 0)

    def __getattr__(self, name):
        if name not in self.allowed_attributes:
            self.forbidden_access_attempts += 1
            raise AttributeError(f'Runtime policy access to {name!r} is forbidden')
        value = getattr(self._environment, name)
        return value.copy() if isinstance(value, np.ndarray) else value


def _snapshot(env, action=None):
    return dict(step=int(env.steps), tool_xyz_m=np.asarray(env.tool_xyz, float).tolist(),
        command_xyz_m=np.asarray(env.command_xyz, float).tolist(),
        ring_xyz_m=np.asarray(env.ring_xyz, float).tolist(),
        bearing_xyz_m=np.asarray(env.bearing_xyz, float).tolist(),
        finger_opening_m=float(env.finger_opening), finger_contacts=int(env.ring_finger_contacts()),
        action=None if action is None else np.asarray(action, float).tolist())


def run_physical_case(env, request, controller, *, scorer):
    """Execute one registered attempt, retaining actual physics and all failures."""
    row = dict(request, attempted=True, success=False, target_score=False, steps=0, trajectory=[],
        failure_reason='', policy_forbidden_access_attempts=0)
    view = ControllerEnvironmentView(env)
    info = {}
    try:
        env.reset(request['ring_xy_m'], request['bearing_xy_m'], seed=request['reset_seed'])
        row['trajectory'].append(_snapshot(env))
        for _ in range(MAX_STEPS):
            if controller.finished:
                break
            action = np.asarray(controller(view), dtype=float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError('Controller must produce a finite four-dimensional action')
            action = np.clip(action, -1., 1.)
            _, done, info = env.step(action)
            row['trajectory'].append(_snapshot(env, action))
            if done:
                break
        # No score or simulator info is visible to the policy or advances it.
        row['target_score'] = bool(scorer(env, request['instruction']))
        row['success'] = bool(row['target_score'] and controller.finished)
        row['failure_reason'] = '' if row['success'] else (
            'controller_finished_without_target_success' if controller.finished else 'timeout')
        row['info'] = dict(info, success=row['success'], failure_reason=row['failure_reason'])
    except Exception as error:
        row['failure_reason'] = 'evaluation_exception'
        row['exception'] = f'{type(error).__name__}: {error}'
        row['info'] = dict(info, success=False, failure_reason='evaluation_exception', exception=row['exception'])
    row['steps'] = len(row['trajectory'])-1 if row['trajectory'] else 0
    row['policy_forbidden_access_attempts'] = view.forbidden_access_attempts
    row['neural_forward_calls'] = int(getattr(controller, 'neural_forward_calls', 0))
    row['waypoint_index'] = int(getattr(controller, 'waypoint_index', 0))
    row['controller_finished'] = bool(getattr(controller, 'finished', False))
    row['input_image_sha256'] = getattr(controller, 'input_image_sha256', None)
    predicted = getattr(controller, 'predicted_waypoints', None)
    row['predicted_waypoints'] = None if predicted is None else np.asarray(predicted, float).tolist()
    return row


def _group_summary(rows, registered):
    lookup = {row['attempt_id']: row for row in rows}
    successes = attempted = target_scores = finished = 0
    failures = Counter()
    for request in registered:
        row = lookup.get(request['attempt_id'])
        if row is None:
            failures['not_executed'] += 1
        else:
            attempted += bool(row.get('attempted'))
            successes += bool(row.get('success'))
            target_scores += bool(row.get('target_score'))
            finished += bool(row.get('controller_finished'))
            if not row.get('success'):
                failures[row.get('failure_reason') or 'unspecified_failure'] += 1
    return dict(episodes=len(registered), attempted_cases=attempted, successes=successes,
        valid_post_execution_targets=target_scores, completed_controllers=finished,
        success_rate=successes/len(registered) if registered else None,
        failure_counts=dict(sorted(failures.items())))


def summarize(rows, schedule):
    registered = {request['attempt_id']: request for request in schedule}
    if len(registered) != len(schedule):
        raise ValueError('Evaluation schedule contains duplicate attempt IDs')
    seen = set()
    for row in rows:
        identifier = row.get('attempt_id')
        if identifier not in registered:
            raise ValueError('Attempt is not registered in the fixed schedule')
        if identifier in seen:
            raise ValueError('Duplicate evaluation attempt')
        if any(row.get(key) != registered[identifier].get(key) for key in (
                'configuration_id', 'suite', 'instruction_index', 'instruction', 'ablation')):
            raise ValueError('Attempt identity differs from the registered schedule')
        seen.add(identifier)
    by_suite = {suite: _group_summary(rows, [r for r in schedule if r['suite'] == suite]) for suite in SUITES}
    by_band = {suite: {str(band): _group_summary(rows,
        [r for r in schedule if r['suite'] == suite and r['distance_band_index'] == band])
        for band in range(3)} for suite in SUITES}
    by_direction = {suite: {str(direction): _group_summary(rows,
        [r for r in schedule if r['suite'] == suite and r['direction_bin_index'] == direction])
        for direction in range(ANGLE_BINS)} for suite in SUITES}
    language = {str(index): _group_summary(rows,
        [r for r in schedule if r['suite'] == 'language_counterfactual' and r['instruction_index'] == index])
        for index in range(3)}
    return dict(by_suite=by_suite, by_distance_band=by_band, by_direction_bin=by_direction,
        language_by_instruction=language,
        inside_language_attempts_are_repeats_of_main=True, language_unique_scenes=24,
        policy_forbidden_access_attempts=sum(r.get('policy_forbidden_access_attempts', 0) for r in rows),
        attempted_rollouts=len(rows), expected_rollouts=len(schedule),
        denominator_policy='All registered attempts, including exceptions, timeouts and unexecuted cases')


def paired_diagnostics(rows):
    """Measure language/vision counterfactuals without selecting model weights."""
    lookup = {(r['suite'], r['configuration_id'], r['instruction_index']): r for r in rows}
    language = dict(complete_pairs=0, same_rgb_for_all_three_commands=0,
        both_side_chunks_changed_from_inside=0, mean_max_abs_waypoint_change_m={})
    repeat = dict(compared=0, identical_trajectories=0, identical_neural_chunks=0,
                  identical_success=0)
    black = dict(compared=0, different_image_hash=0, different_neural_chunks=0,
                 main_success_and_black_failure=0)
    side_changes = {'left': [], 'right': []}
    identifiers = sorted({r['configuration_id'] for r in rows})
    for identifier in identifiers:
        main = lookup.get(('main', identifier, 0))
        blackout = lookup.get(('black_image', identifier, 0))
        commands = [lookup.get(('language_counterfactual', identifier, index)) for index in range(3)]
        if main is not None and blackout is not None:
            black['compared'] += 1
            black['different_image_hash'] += bool(main.get('input_image_sha256')
                and main.get('input_image_sha256') != blackout.get('input_image_sha256'))
            black['different_neural_chunks'] += bool(main.get('predicted_waypoints') is not None
                and main.get('predicted_waypoints') != blackout.get('predicted_waypoints'))
            black['main_success_and_black_failure'] += bool(main.get('success') and not blackout.get('success'))
        if main is not None and commands[0] is not None:
            repeated = commands[0]
            repeat['compared'] += 1
            repeat['identical_trajectories'] += main.get('trajectory') == repeated.get('trajectory')
            repeat['identical_neural_chunks'] += main.get('predicted_waypoints') == repeated.get('predicted_waypoints')
            repeat['identical_success'] += main.get('success') == repeated.get('success')
        if all(r is not None for r in commands):
            language['complete_pairs'] += 1
            hashes = [r.get('input_image_sha256') for r in commands]
            language['same_rgb_for_all_three_commands'] += bool(hashes[0] and len(set(hashes)) == 1)
            chunks = [r.get('predicted_waypoints') for r in commands]
            if all(chunk is not None for chunk in chunks):
                chunks = [np.asarray(chunk, dtype=float) for chunk in chunks]
                if len({chunk.shape for chunk in chunks}) != 1:
                    raise ValueError('Counterfactual neural waypoint shapes differ')
                changes = [float(np.max(np.abs(chunk[:, :3]-chunks[0][:, :3]))) for chunk in chunks[1:]]
                language['both_side_chunks_changed_from_inside'] += all(change > 1e-8 for change in changes)
                for name, change in zip(('left', 'right'), changes):
                    side_changes[name].append(change)
    language['mean_max_abs_waypoint_change_m'] = {
        name: float(np.mean(changes)) if changes else None for name, changes in side_changes.items()}
    return dict(language=language, inside_repeat=repeat, black_image=black,
        interpretation='Paired observations and action changes; no unseen-language or real-camera claim')


def run(manifest_path, receipt_path, destination, *, device='cpu'):
    """Run only the frozen checkpoint, using no true-state controller inputs."""
    if Path(destination).exists():
        raise FileExistsError('Frozen evaluation output already exists; choose a separate reproduction destination')
    manifest, receipt = _read_json(manifest_path), _read_json(receipt_path)
    validate_receipt(receipt, manifest)
    from .vla_control import VLAController
    from .vla_env import VLAEnv, score_target
    from .vla_policy import load_vla_policy
    import torch
    torch.set_num_threads(2)
    model, metadata = load_vla_policy(receipt['checkpoint_path'], device=device)
    validate_checkpoint_metadata(metadata, manifest)
    env = VLAEnv(max_steps=MAX_STEPS)
    rows = []
    schedule = evaluation_schedule(manifest['configurations'])
    try:
        for index, request in enumerate(schedule):
            controller = VLAController(model, request['instruction'], ablation=request['ablation'])
            rows.append(run_physical_case(env, request, controller, scorer=score_target))
            if (index+1) % 12 == 0:
                print(f'V5 frozen evaluation {index+1}/{len(schedule)}', flush=True)
    finally:
        env.close()
    report = dict(schema_version=1, manifest_sha256=sha256_file(manifest_path),
        model_freeze_receipt_sha256=sha256_file(receipt_path), checkpoint_sha256=receipt['checkpoint_sha256'],
        summary=summarize(rows, schedule), paired_diagnostics=paired_diagnostics(rows), rollouts=rows,
        runtime_policy_boundary='Restricted RGB/own-proprioception API; true-state grading after execution',
        object_oracle_used_for_actions=False, model_selection_from_test=False,
        success_definition='controller finished and post-execution target score valid',
        checkpoint_metadata=metadata, inference_device=device)
    _write_new_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--dataset-manifest', required=True)
    prepare_parser.add_argument('--out', required=True)
    prepare_parser.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    freeze_parser = commands.add_parser('freeze')
    freeze_parser.add_argument('--manifest', required=True)
    freeze_parser.add_argument('--checkpoint', required=True)
    freeze_parser.add_argument('--training-report', required=True)
    freeze_parser.add_argument('--out', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--manifest', required=True)
    run_parser.add_argument('--receipt', required=True)
    run_parser.add_argument('--out', required=True)
    run_parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.dataset_manifest, args.out, protocol_path=args.protocol)
        print(json.dumps(dict(cases=result['expected_cases'], seed=result['seed'])))
    elif args.command == 'freeze':
        result = freeze_model_receipt(args.manifest, args.checkpoint, args.training_report, args.out)
        print(json.dumps(dict(checkpoint_sha256=result['checkpoint_sha256'], frozen_utc=result['frozen_utc'])))
    else:
        result = run(args.manifest, args.receipt, args.out, device=args.device)
        print(json.dumps(result['summary']['by_suite']))


if __name__ == '__main__':
    main()
