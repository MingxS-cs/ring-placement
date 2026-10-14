import copy
import json
from pathlib import Path

import numpy as np
import pytest


def test_fresh_geometry_and_paired_schedule_preserve_all_strata():
    from ring_learning.motion_ablation import scene_configurations, evaluation_schedule
    from ring_learning.vla_evaluate import scene_configurations as old_cases
    cases = scene_configurations()
    assert cases == scene_configurations()
    assert len(cases) == 96
    old_positions = {(tuple(c['ring_xy_m']), tuple(c['bearing_xy_m'])) for c in old_cases()}
    assert not old_positions.intersection((tuple(c['ring_xy_m']), tuple(c['bearing_xy_m'])) for c in cases)
    for band in range(3):
        for direction in range(8):
            group = [c for c in cases if c['distance_band_index'] == band and c['direction_bin_index'] == direction]
            assert {c['position_index'] for c in group} == {0, 1, 2, 3}
    schedule = evaluation_schedule(cases)
    assert len(schedule) == len({r['attempt_id'] for r in schedule}) == 192
    for first, second in zip(schedule[::2], schedule[1::2]):
        assert (first['variant'], second['variant']) == ('video_motion', 'straight_transfer')
        for key in ('configuration_id', 'ring_xy_m', 'bearing_xy_m', 'reset_seed', 'instruction'):
            assert first[key] == second[key]
    with pytest.raises(ValueError):
        scene_configurations(seed=20261024)
    with pytest.raises(ValueError):
        evaluation_schedule(cases[:-1])


def test_shared_initial_image_is_copied_and_proprioception_still_uses_restricted_view():
    from ring_learning.motion_ablation import TracedController
    from ring_learning.vla_evaluate import ControllerEnvironmentView
    class Environment:
        tool_xyz = np.array([.45, .01, .16])
        def render_policy(self):
            raise AssertionError('Policy must use the captured shared image')
    class Delegate:
        waypoint_index = 4
        def __call__(self, view):
            first = view.render_policy()
            first[:] = 99
            self.observed = view.render_policy()
            np.testing.assert_array_equal(view.tool_xyz, [.45, .01, .16])
            with pytest.raises(AttributeError):
                _ = view.ring_xyz
            return np.zeros(4)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    delegate = Delegate()
    traced = TracedController(delegate, initial_rgb=frame)
    frame[:] = 17
    restricted = ControllerEnvironmentView(Environment())
    traced(restricted)
    assert not delegate.observed.any()
    assert restricted.forbidden_access_attempts == 1
    assert traced.controller_waypoint_indices == [4]


def test_observation_capture_failure_stays_a_failed_registered_attempt():
    from ring_learning.motion_ablation import TracedController
    class Delegate:
        waypoint_index = 0
        def __call__(self, view):
            raise AssertionError('Failed capture must not retry model inference')
    with pytest.raises(RuntimeError, match='capture'):
        TracedController(Delegate(), observation_error='capture failed')(object())


@pytest.mark.parametrize('kind', ['checkpoint', 'dataset', 'source', 'protocol'])
def test_every_bound_file_rejects_tampering(tmp_path, kind):
    from ring_learning.motion_ablation import file_binding, check_bindings
    path = tmp_path/f'{kind}.txt'
    path.write_text('original\n', encoding='utf-8')
    bindings = {kind: file_binding(path, source=(kind == 'source'))}
    check_bindings(bindings)
    path.write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match=kind):
        check_bindings(bindings)


def test_prepare_reads_only_train_npz_and_refuses_existing_output(tmp_path, monkeypatch):
    from ring_learning.motion_ablation import prepare, validate_manifest, validate_freeze_receipt
    reads = []
    load = np.load
    def tracked(path, *args, **kwargs):
        reads.append(Path(path).name)
        assert Path(path).name == 'train.npz', 'counterfactual fit must not open val/test arrays'
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', tracked)
    out = tmp_path/'experiment'
    manifest = prepare(out)
    assert reads == ['train.npz']
    assert len(manifest['configurations']) == 96 and len(manifest['schedule']) == 192
    assert manifest['shared_weights']['unchanged_parameters_equal']
    assert manifest['shared_weights']['common_readout_rows_equal']
    assert Path(manifest['models']['straight_transfer']['path']).exists()
    from ring_learning.vla_policy import load_vla_policy
    _, baseline_metadata = load_vla_policy(manifest['models']['straight_transfer']['path'])
    assert 'validation_metrics' not in baseline_metadata
    assert 'validation_metrics' in baseline_metadata['inherited_v5_metadata']
    validate_manifest(manifest)
    receipt = json.loads((out/'freeze_receipt.json').read_text())
    validate_freeze_receipt(receipt, out/'manifest.json')
    with pytest.raises(FileExistsError):
        prepare(out)
    changed = copy.deepcopy(manifest)
    changed['configurations'][0]['ring_xy_m'][0] += .001
    with pytest.raises(ValueError, match='scene|geometry'):
        validate_manifest(changed)
    Path(manifest['models']['straight_transfer']['path']).write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='straight_transfer'):
        validate_manifest(manifest)


def test_shared_parameter_audit_rejects_vision_or_critical_action_changes():
    import torch
    from ring_learning.motion_ablation import shared_parameter_audit
    from ring_learning.vla_policy import TinyVLA
    base = TinyVLA()
    straight = copy.deepcopy(base)
    with torch.no_grad():
        straight.coefficient_head[-1].bias[160] += 1.
    shared_parameter_audit(base, straight)
    with torch.no_grad():
        straight.coefficient_head[-1].bias[180] += 1.
    with pytest.raises(ValueError, match='shared'):
        shared_parameter_audit(base, straight)
    straight = copy.deepcopy(base)
    with torch.no_grad():
        straight.vision[0].weight[0, 0, 0, 0] += 1.
    with pytest.raises(ValueError, match='shared'):
        shared_parameter_audit(base, straight)


def test_existing_evaluation_is_never_overwritten(tmp_path):
    from ring_learning.motion_ablation import run
    output = tmp_path/'evaluation.json'
    output.write_bytes(b'preserved report')
    with pytest.raises(FileExistsError):
        run(tmp_path/'missing_manifest.json', output)
    assert output.read_bytes() == b'preserved report'


class FakeController:
    def __init__(self):
        self.waypoint_index = 4
        self.finished = False
        self.neural_forward_calls = 1
        self.predicted_waypoints = np.zeros((23, 4))
        self.input_image_sha256 = 'a'*64
    def __call__(self, env):
        self.waypoint_index += 1
        self.finished = self.waypoint_index == 6
        return np.array([1., 0., 0., -1.])


class FakeEnvironment:
    def reset(self, ring, bearing, seed):
        self.steps = 0
        self.command_xyz = np.r_[ring, .16]
        self.tool_xyz = self.command_xyz.copy()
        self.ring_xyz = np.r_[ring, .16]
        self.bearing_xyz = np.r_[bearing, .1025]
        self.finger_opening = .00025
    def ring_finger_contacts(self):
        return 1
    def step(self, action):
        self.command_xyz += .003*action[:3]
        self.tool_xyz = self.command_xyz.copy()
        self.ring_xyz += .003*action[:3]
        self.steps += 1
        return np.zeros(7), self.steps == 2, {}


def test_traced_controller_does_not_change_actions_and_records_pre_stage():
    from ring_learning.motion_ablation import TracedController
    delegate = FakeController()
    traced = TracedController(delegate)
    np.testing.assert_array_equal(traced(object()), [1., 0., 0., -1.])
    np.testing.assert_array_equal(traced(object()), [1., 0., 0., -1.])
    assert traced.controller_waypoint_indices == [4, 5]
    assert delegate.waypoint_index == traced.waypoint_index == 6
    assert traced.finished


def test_physical_runner_retains_actions_actual_paths_and_execution_errors():
    from ring_learning.motion_ablation import evaluation_schedule, scene_configurations, run_case
    request = evaluation_schedule(scene_configurations())[0]
    row = run_case(FakeEnvironment(), request, FakeController(), scorer=lambda env, text: True)
    assert row['success'] and row['target_score']
    assert row['controller_waypoint_indices'] == [4, 5]
    assert row['transfer_metrics']['transfer_steps'] == 2
    assert row['transfer_metrics']['ring']['path_length_m'] == pytest.approx(.006)
    assert row['transfer_metrics']['ring_airborne_fraction'] == 1.
    class BrokenEnvironment(FakeEnvironment):
        def reset(self, *args, **kwargs):
            raise ValueError('broken reset')
    broken = run_case(BrokenEnvironment(), request, FakeController(), scorer=lambda env, text: True)
    assert broken['attempted'] and not broken['success']
    assert broken['failure_reason'] == 'evaluation_exception'
    assert broken['transfer_metrics']['transfer_steps'] == 0
    assert broken['transfer_metrics']['ring']['degenerate']


def test_path_metrics_use_actual_samples_and_keep_degenerate_paths():
    from ring_learning.motion_ablation import path_metrics, transfer_metrics
    curve = np.array([[0., 0.], [.05, .015], [.1, 0.]])
    measured = path_metrics(curve)
    assert measured['max_abs_lateral_m'] == pytest.approx(.015)
    assert measured['path_length_m'] == pytest.approx(2*np.hypot(.05, .015))
    assert measured['arc_to_chord_ratio'] > 1.
    stationary = path_metrics(np.repeat([[0., 0.]], 3, axis=0))
    assert stationary['degenerate'] and stationary['arc_to_chord_ratio'] is None
    trajectory = []
    for x, y in ((-.1, 0.), (0., 0.), (.05, .015), (.1, 0.), (.2, 0.)):
        trajectory.append(dict(ring_xyz_m=[x, y, .16], tool_xyz_m=[x, y, .161],
            command_xyz_m=[x, y, .161], bearing_xyz_m=[.1, 0., .1025], finger_contacts=1))
    row = dict(trajectory=trajectory, controller_waypoint_indices=[3, 4, 19, 20],
        predicted_waypoints=np.zeros((23, 4)).tolist())
    result = transfer_metrics(row)
    assert result['transfer_steps'] == 2
    assert result['transfer_start_trajectory_index'] == 1
    assert result['transfer_end_trajectory_index'] == 3
    assert result['ring']['max_abs_lateral_m'] == pytest.approx(.015)
    assert result['terminal_center_error_m'] == pytest.approx(.1)


def test_summary_keeps_full_denominators_and_pairs_shared_critical_predictions():
    from ring_learning.motion_ablation import scene_configurations, evaluation_schedule, summarize
    schedule = evaluation_schedule(scene_configurations())
    chunks = np.zeros((23, 4))
    first = dict(schedule[0], attempted=True, success=True, target_score=True,
        controller_finished=True, failure_reason='', input_image_sha256='a'*64,
        predicted_waypoints=chunks.tolist(), transfer_metrics={}, trajectory=[])
    other = chunks.copy()
    other[5, 0] = .01
    second = dict(schedule[1], attempted=True, success=False, target_score=False,
        controller_finished=True, failure_reason='controller_finished_without_target_success',
        input_image_sha256='a'*64, predicted_waypoints=other.tolist(),
        transfer_metrics={}, trajectory=[])
    summary = summarize([first, second], schedule)
    assert summary['by_variant']['video_motion']['episodes'] == 96
    assert summary['by_variant']['video_motion']['successes'] == 1
    assert summary['by_variant']['video_motion']['failure_counts'] == {'not_executed': 95}
    assert summary['by_variant']['straight_transfer']['failure_counts'] == {
        'controller_finished_without_target_success': 1, 'not_executed': 95}
    assert summary['by_distance_band']['video_motion']['0']['episodes'] == 32
    assert summary['by_direction_bin']['video_motion']['0']['episodes'] == 12
    assert summary['paired']['complete_pairs'] == 1
    assert summary['paired']['same_initial_rgb'] == 1
    assert summary['paired']['common_predicted_outputs_equal'] == 1
    assert summary['paired']['changed_transfer_predictions'] == 1
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        summarize([first, first], schedule)
    critical_bad = copy.deepcopy(second)
    critical_bad['predicted_waypoints'][1][0] += .001
    mismatch = summarize([first, critical_bad], schedule)
    assert mismatch['paired']['common_predicted_outputs_equal'] == 0
    assert mismatch['paired']['maximum_common_prediction_difference'] == pytest.approx(.001)
    assert mismatch['paired']['common_prediction_mismatches'] == 1
    reset_bad = copy.deepcopy(first)
    reset_bad['reset_seed'] += 1
    with pytest.raises(ValueError, match='identity|schedule'):
        summarize([reset_bad, second], schedule)
