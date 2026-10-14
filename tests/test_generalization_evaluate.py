import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


DEVELOPMENT = 'data/v4_development_annotations.json'
PROTOCOL = 'docs/superpowers/specs/2026-10-14-generalization-design.md'


def prepared(tmp_path):
    from ring_learning.generalization_evaluate import prepare
    destination = tmp_path/'configurations.json'
    return prepare(DEVELOPMENT, destination, protocol_path=PROTOCOL), destination


def test_frozen_scenes_have_exact_strata_and_conditional_workspace_geometry(tmp_path):
    manifest, _ = prepared(tmp_path)
    cases = manifest['configurations']
    assert len(cases) == 96
    assert len({c['configuration_id'] for c in cases}) == 96
    assert manifest['seed'] == 20261020
    for band in range(3):
        for angle_bin in range(8):
            cell = [c for c in cases if c['distance_band_index'] == band and c['direction_bin_index'] == angle_bin]
            assert len(cell) == 4
            assert {c['position_index'] for c in cell} == {0, 1, 2, 3}
    for case in cases:
        ring, bearing = np.array(case['ring_xy_m']), np.array(case['bearing_xy_m'])
        delta = bearing-ring
        distance = np.linalg.norm(delta)
        lower, upper = case['distance_band_m']
        assert lower <= distance <= upper
        assert np.all(bearing >= [.40, -.10]) and np.all(bearing <= [.52, .10])
        assert np.all(ring >= [.32, -.23]) and np.all(ring <= [.62, .23])
        np.testing.assert_allclose(delta, case['bearing_minus_ring_xy_m'], atol=1e-14)
        assert case['angle_rad'] == pytest.approx(np.arctan2(delta[1], delta[0]))
        assert not any(k in case for k in ('source_episode_id', 'episode_id'))


def test_scene_draw_is_reproducible_and_manifest_cannot_be_overwritten(tmp_path):
    from ring_learning.generalization_evaluate import prepare
    first, path = prepared(tmp_path)
    second = prepare(DEVELOPMENT, tmp_path/'other.json', protocol_path=PROTOCOL)
    assert first == second
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(DEVELOPMENT, path, protocol_path=PROTOCOL)
    assert path.read_bytes() == saved


def test_manifest_rejects_changed_controller_environment_or_development(tmp_path):
    from ring_learning.generalization_evaluate import validate_manifest
    manifest, _ = prepared(tmp_path)
    validate_manifest(manifest, DEVELOPMENT)
    for key in ('controller_source_hash', 'development_annotations_sha256'):
        wrong = copy.deepcopy(manifest)
        wrong[key] = '0'*64
        with pytest.raises(ValueError):
            validate_manifest(wrong, DEVELOPMENT)
    wrong = copy.deepcopy(manifest)
    wrong['environment']['sim_sha256'] = '0'*64
    with pytest.raises(ValueError, match='environment'):
        validate_manifest(wrong, DEVELOPMENT)


class FakeEnvironment:
    max_steps = 2
    def reset(self, ring, bearing, seed):
        self.steps, self.phase = 0, 4
        self.ring_xyz = np.array([.02, .01, .0035])
        self.command_xyz = np.array([0., 0., .16])
        self.tool_xyz = np.array([-.003, 0., .16])
        self.actions = []
        return np.zeros(30)
    def ring_finger_contacts(self):
        return 1
    def step(self, action):
        self.actions.append(action.copy())
        self.command_xyz += .003*action[:3]
        self.tool_xyz = self.command_xyz-[.003, 0., 0.]
        self.ring_xyz += [.001, 0., 0.]
        self.steps += 1
        return np.zeros(30), self.steps == self.max_steps, dict(failure_reason='timeout_transfer', phase_index=4)
    def success(self):
        return False


class FakeController:
    progress = curve_length_m = arc_cursor_m = 0.
    learned_reference_evaluations = transfer_steps = planned_steps = lag_pause_steps = 0
    def __call__(self, env, obs):
        self.progress += .5
        self.transfer_steps += 1
        return np.array([1.25, 0., 0., -1.])


def test_rollout_saves_physical_ring_command_and_executed_actions_separately():
    from ring_learning.generalization_evaluate import run_physical_case
    env = FakeEnvironment()
    case = dict(configuration_id='case', ring_xy_m=[0, 0], bearing_xy_m=[.1, 0], reset_seed=1,
                distance_band_index=0, direction_bin_index=0, position_index=0)
    row = run_physical_case(env, case, FakeController(), 'v4')
    assert not row['success'] and row['steps'] == 2
    assert row['info']['failure_reason'] == 'timeout_transfer'
    assert len(row['trajectory']) == 3
    final = row['trajectory'][-1]
    np.testing.assert_allclose(final['ring_xyz_m'], [.022, .01, .0035])
    np.testing.assert_allclose(final['command_xyz_m'], [.006, 0., .16])
    np.testing.assert_array_equal(final['action'], env.actions[-1])
    assert final['action'][0] == 1.
    assert row['transfer_contact_steps'] == 2
    assert row['transfer_tool_command_mean_lag_m'] == pytest.approx(.003)
    assert not np.allclose(row['transfer_ring_xy_m'], row['transfer_command_xy_m'])


def test_reset_errors_remain_failed_attempts():
    from ring_learning.generalization_evaluate import run_physical_case
    class Broken(FakeEnvironment):
        def reset(self, *args):
            raise ValueError('bad reset')
    cfg = dict(configuration_id='bad', ring_xy_m=[0, 0], bearing_xy_m=[.1, 0], reset_seed=1,
               distance_band_index=0, direction_bin_index=0, position_index=0)
    row = run_physical_case(Broken(), cfg, FakeController(), 'v4')
    assert row['attempted'] and not row['success'] and row['steps'] == 0
    assert row['info']['failure_reason'] == 'evaluation_exception'
    assert 'bad reset' in row['info']['exception']


def test_summaries_keep_all_96_denominators_and_stratum_failures(tmp_path):
    from ring_learning.generalization_evaluate import summarize
    manifest, _ = prepared(tmp_path)
    cases = manifest['configurations']
    rows = []
    for index, success in enumerate((True, False)):
        rows.append(dict(controller='v4', configuration_id=cases[index]['configuration_id'],
            success=success, attempted=True, steps=1, info={} if success else {'failure_reason':'failed_grasp'},
            distance_band_index=cases[index]['distance_band_index'], direction_bin_index=cases[index]['direction_bin_index'],
            learned_reference_evaluations=0, transfer_steps=0, planned_steps=1, lag_pause_steps=0,
            transfer_contact_steps=0, transfer_tool_command_mean_lag_m=None, transfer_tool_command_max_lag_m=None))
    report = summarize(rows, cases)
    total = report['overall']['v4']
    assert total['episodes'] == 96 and total['attempted_cases'] == 2
    assert total['successes'] == 1 and total['success_rate'] == 1/96
    assert total['failure_counts'] == {'failed_grasp':1, 'not_executed':94}
    assert report['by_distance_band']['v4']['0']['episodes'] == 32
    assert report['by_direction_bin']['v4']['0']['episodes'] == 12
    assert report['overall']['straight']['success_rate'] == 0.


def checkpoint_and_blind_receipt(tmp_path, manifest):
    from ring_learning.generalization_policy import DirectionRidgePolicy, save_generalization_policy, source_signature
    from ring_learning.generalization_evaluate import training_protocol
    policy = tmp_path/'policy.json'
    metadata = dict(environment=manifest['environment'], implementation_sources=source_signature(),
        original_video_sha256=manifest['source_video_sha256'], real_episodes_sha256=manifest['real_episodes_sha256'],
        motion_sha256=manifest['real_motion_sha256'], development_annotations_sha256=manifest['development_annotations_sha256'],
        training_sources=manifest['source_splits']['train'], validation_sources=manifest['source_splits']['val'],
        selection=dict(candidate_grid=training_protocol()['candidate_grid'], validation_used_for_selection=False))
    save_generalization_policy(policy, DirectionRidgePolicy(np.zeros((3,2)), [0.,0.], [1.,1.]), metadata)
    blind = tmp_path/'blind_receipt.json'
    blind.write_text(json.dumps(dict(annotation_sha256='e'*64, source_sha256=manifest['source_video_sha256'],
        endpoint_manifest_sha256=manifest['real_episodes_sha256'], existing_motion_sha256=manifest['real_motion_sha256'],
        reserved_source_ids=['E20','E21','E22'], valid_episodes=3, measured_samples=35)), encoding='utf-8')
    return policy, blind


def test_freeze_receipt_binds_model_protocol_scenes_and_metadata_only_holdout(tmp_path):
    from ring_learning.generalization_evaluate import freeze_model_receipt
    manifest, manifest_path = prepared(tmp_path)
    checkpoint, blind = checkpoint_and_blind_receipt(tmp_path, manifest)
    destination = tmp_path/'model_freeze.json'
    receipt = freeze_model_receipt(manifest_path, checkpoint, DEVELOPMENT, destination, blind_receipt=blind)
    assert receipt['selected_checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert receipt['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert receipt['holdout_annotations_sha256'] == 'e'*64
    assert receipt['holdout_sources'] == ['E20','E21','E22']
    assert receipt['training_protocol_sha256'] == hashlib.sha256(Path(PROTOCOL).read_bytes()).hexdigest()
    assert set(receipt['training_sources']).isdisjoint(receipt['holdout_sources'])
    assert receipt['frozen_before_holdout_unseal']
    assert receipt['frozen_utc']
    with pytest.raises(FileExistsError):
        freeze_model_receipt(manifest_path, checkpoint, DEVELOPMENT, destination, blind_receipt=blind)


def test_freeze_rejects_holdout_source_leakage_before_writing_receipt(tmp_path):
    from ring_learning.generalization_evaluate import freeze_model_receipt
    manifest, manifest_path = prepared(tmp_path)
    checkpoint, blind = checkpoint_and_blind_receipt(tmp_path, manifest)
    value = json.loads(blind.read_text())
    value['reserved_source_ids'] = ['E01', 'E21', 'E22']
    blind.write_text(json.dumps(value), encoding='utf-8')
    destination = tmp_path/'model_freeze.json'
    with pytest.raises(ValueError, match='holdout|reserved|source'):
        freeze_model_receipt(manifest_path, checkpoint, DEVELOPMENT, destination, blind_receipt=blind)
    assert not destination.exists()
