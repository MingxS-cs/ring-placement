import copy
import hashlib
import json

import numpy as np
import pytest


def test_new_scene_seed_reproducibly_covers_all_predeclared_strata():
    from ring_learning.vla_evaluate import scene_configurations
    first = scene_configurations()
    assert first == scene_configurations()
    assert len(first) == 96
    assert len({c['configuration_id'] for c in first}) == 96
    for band in range(3):
        for angle in range(8):
            group = [c for c in first if c['distance_band_index'] == band and c['direction_bin_index'] == angle]
            assert {c['position_index'] for c in group} == {0, 1, 2, 3}
    for case in first:
        ring, bearing = np.array(case['ring_xy_m']), np.array(case['bearing_xy_m'])
        lower, upper = case['distance_band_m']
        assert lower <= np.linalg.norm(bearing-ring) <= upper
        assert np.all(ring >= [.32, -.23]) and np.all(ring <= [.62, .23])
        assert np.all(bearing >= [.40, -.10]) and np.all(bearing <= [.52, .10])
        assert not {'source_episode_id', 'episode_id'}.intersection(case)
    with pytest.raises(ValueError):
        scene_configurations(seed=20261020)


def test_locked_schedule_reports_language_control_repeats_and_full_denominators():
    from ring_learning.vla_evaluate import evaluation_schedule, scene_configurations
    cases = scene_configurations()
    rows = evaluation_schedule(cases)
    assert len(rows) == 264
    main = [r for r in rows if r['suite'] == 'main']
    black = [r for r in rows if r['suite'] == 'black_image']
    language = [r for r in rows if r['suite'] == 'language_counterfactual']
    assert len(main) == len(black) == 96
    assert len(language) == 72
    assert sum(r['repeat_of_main'] for r in language) == 24
    chosen = {cases[i]['configuration_id'] for i in range(0, 96, 4)}
    for index in range(3):
        assert {r['configuration_id'] for r in language if r['instruction_index'] == index} == chosen
    assert all(r['ablation'] == 'black_image' for r in black)
    assert all(r['instruction_index'] == 0 for r in main+black)


def bound_manifest(tmp_path):
    from ring_learning.vla_evaluate import prepare
    dataset = tmp_path/'dataset.json'
    protocol = tmp_path/'protocol.md'
    source = tmp_path/'implementation.py'
    files = {}
    for name in ('train.npz', 'val.npz'):
        path = tmp_path/name
        path.write_bytes(name.encode())
        files[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    dataset.write_text(json.dumps({'schema_version': 1, 'source_video_sha256': 'a'*64,
        'files': files, 'split_sources': {'train': ['E01'], 'val': ['E05']}}), encoding='utf-8')
    protocol.write_text('Locked epoch selection: validation only.', encoding='utf-8')
    source.write_text('value = 1\n', encoding='utf-8')
    destination = tmp_path/'configurations.json'
    manifest = prepare(dataset, destination, protocol_path=protocol,
                       implementation_paths={'sample': source})
    return manifest, destination, dataset, protocol, source


def test_manifest_cannot_be_overwritten_and_binds_code_data_and_protocol(tmp_path):
    from ring_learning.vla_evaluate import prepare, validate_manifest
    manifest, destination, dataset, protocol, source = bound_manifest(tmp_path)
    validate_manifest(manifest)
    saved = destination.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(dataset, destination, protocol_path=protocol, implementation_paths={'sample': source})
    assert destination.read_bytes() == saved
    changed = copy.deepcopy(manifest)
    changed['configurations'][0]['ring_xy_m'][0] += .001
    with pytest.raises(ValueError, match='geometry|scene'):
        validate_manifest(changed)
    source.write_text('value = 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source|implementation'):
        validate_manifest(manifest)


def test_selected_checkpoint_receipt_is_immutable_and_rejects_changed_weights(tmp_path):
    from ring_learning.vla_evaluate import freeze_model_receipt, validate_receipt
    manifest, manifest_path, _, _, _ = bound_manifest(tmp_path)
    checkpoint = tmp_path/'policy.pt'
    checkpoint.write_bytes(b'test checkpoint')
    training = tmp_path/'training.json'
    training.write_text('{"metadata": {"selection": {"selected_epoch": 3, "split": "validation only"}}}', encoding='utf-8')
    receipt_path = tmp_path/'freeze.json'
    receipt = freeze_model_receipt(manifest_path, checkpoint, training, receipt_path)
    assert receipt['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert receipt['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert receipt['frozen_before_test_execution'] is True
    assert receipt['selected_epoch'] == 3
    assert receipt['selection_record']['split'] == 'validation only'
    validate_receipt(receipt, manifest)
    with pytest.raises(FileExistsError):
        freeze_model_receipt(manifest_path, checkpoint, training, receipt_path)
    checkpoint.write_bytes(b'new checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        validate_receipt(receipt, manifest)


def test_dataset_arrays_are_bound_and_changed_training_labels_reject_evaluation(tmp_path):
    from ring_learning.vla_evaluate import validate_manifest
    manifest, _, _, _, _ = bound_manifest(tmp_path)
    (tmp_path/'train.npz').write_bytes(b'changed training data')
    with pytest.raises(ValueError, match='dataset|arrays'):
        validate_manifest(manifest)


def test_selected_model_metadata_cannot_substitute_dataset_or_source_membership(tmp_path):
    from ring_learning.vla_evaluate import validate_checkpoint_metadata
    manifest, _, _, _, _ = bound_manifest(tmp_path)
    metadata = dict(dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
        train_npz_sha256=manifest['dataset_files']['train.npz']['sha256'],
        val_npz_sha256=manifest['dataset_files']['val.npz']['sha256'],
        training_sources=['E01'], validation_sources=['E05'], implementation_sources={})
    validate_checkpoint_metadata(metadata, manifest)
    wrong = copy.deepcopy(metadata)
    wrong['validation_sources'] = ['E01']
    with pytest.raises(ValueError, match='source|membership'):
        validate_checkpoint_metadata(wrong, manifest)
    wrong = copy.deepcopy(metadata)
    wrong['train_npz_sha256'] = '0'*64
    with pytest.raises(ValueError, match='training|dataset'):
        validate_checkpoint_metadata(wrong, manifest)


class FakeEnvironment:
    max_steps = 2
    def reset(self, ring, bearing, seed):
        self.steps = 0
        self.command_xyz = np.array([.4, 0., .16])
        self.tool_xyz = self.command_xyz.copy()
        self.ring_xyz = np.r_[ring, .1035]
        self.bearing_xyz = np.r_[bearing, .1025]
        self.finger_opening = .006
        self.score_calls = 0
        self.actions = []
    def observe(self):
        return np.r_[self.tool_xyz, self.command_xyz-self.tool_xyz, self.finger_opening]
    def render_policy(self):
        return np.full((256, 256, 3), 127, dtype=np.uint8)
    def ring_finger_contacts(self):
        return 1
    def step(self, action):
        self.actions.append(action.copy())
        self.command_xyz += .003*action[:3]
        self.tool_xyz = self.command_xyz-[.001, 0., 0.]
        self.ring_xyz += [.001, 0., 0.]
        self.steps += 1
        return self.observe(), self.steps >= self.max_steps, {'failure_reason': 'timeout'}


class FakeController:
    neural_forward_calls = 1
    waypoint_index = 0
    finished = False
    def __call__(self, env):
        assert env.observe().shape == (7,)
        assert env.render_policy().shape == (256, 256, 3)
        self.waypoint_index += 1
        return np.array([1.25, 0., 0., -1.])


def scorer(env, instruction):
    env.score_calls += 1
    return False


def request():
    from ring_learning.vla_evaluate import evaluation_schedule, scene_configurations
    return evaluation_schedule(scene_configurations())[0]


def test_rollout_preserves_executed_physics_and_only_scores_after_controller():
    from ring_learning.vla_evaluate import run_physical_case
    env = FakeEnvironment()
    row = run_physical_case(env, request(), FakeController(), scorer=scorer)
    assert row['attempted'] and not row['success']
    assert row['steps'] == 2 and env.score_calls == 1
    assert row['policy_forbidden_access_attempts'] == 0
    assert len(row['trajectory']) == 3
    assert row['neural_forward_calls'] == 1
    np.testing.assert_array_equal(row['trajectory'][-1]['action'], env.actions[-1])
    assert env.actions[-1][0] == 1.
    assert not np.allclose(row['trajectory'][-1]['ring_xyz_m'], row['trajectory'][-1]['command_xyz_m'])


def test_forbidden_oracle_access_is_blocked_and_retained_as_failed_attempt():
    from ring_learning.vla_evaluate import run_physical_case
    class OracleController(FakeController):
        def __call__(self, env):
            return env.ring_xyz
    row = run_physical_case(FakeEnvironment(), request(), OracleController(), scorer=scorer)
    assert not row['success'] and row['attempted'] and row['steps'] == 0
    assert row['policy_forbidden_access_attempts'] == 1
    assert row['failure_reason'] == 'evaluation_exception'
    assert 'ring_xyz' in row['exception']


def test_target_score_at_timeout_does_not_count_as_complete_task_success():
    from ring_learning.vla_evaluate import run_physical_case
    row = run_physical_case(FakeEnvironment(), request(), FakeController(), scorer=lambda env, instruction: True)
    assert row['target_score'] is True
    assert row['controller_finished'] is False
    assert row['success'] is False
    assert row['failure_reason'] == 'timeout'


def test_complete_controller_and_valid_target_are_both_required_for_success():
    from ring_learning.vla_evaluate import run_physical_case
    class CompleteController(FakeController):
        def __call__(self, env):
            result = super().__call__(env)
            if self.waypoint_index == 2:
                self.finished = True
            return result
    row = run_physical_case(FakeEnvironment(), request(), CompleteController(), scorer=lambda env, instruction: True)
    assert row['target_score'] and row['controller_finished'] and row['success']
    assert row['failure_reason'] == ''


def test_reset_errors_and_unexecuted_cases_remain_in_full_schedule_denominators():
    from ring_learning.vla_evaluate import evaluation_schedule, run_physical_case, scene_configurations, summarize
    class BrokenEnvironment(FakeEnvironment):
        def reset(self, *args, **kwargs):
            raise ValueError('reset failed')
    row = run_physical_case(BrokenEnvironment(), request(), FakeController(), scorer=scorer)
    assert row['attempted'] and not row['success'] and row['steps'] == 0
    assert 'reset failed' in row['exception']
    schedule = evaluation_schedule(scene_configurations())
    report = summarize([row], schedule)
    main = report['by_suite']['main']
    assert main['episodes'] == 96 and main['attempted_cases'] == 1
    assert main['failure_counts'] == {'evaluation_exception': 1, 'not_executed': 95}
    assert report['by_suite']['black_image']['episodes'] == 96
    assert report['by_suite']['language_counterfactual']['episodes'] == 72
    assert report['by_distance_band']['main']['0']['episodes'] == 32
    assert report['by_direction_bin']['main']['0']['episodes'] == 12


def test_summary_rejects_duplicate_or_unregistered_attempts():
    from ring_learning.vla_evaluate import evaluation_schedule, scene_configurations, summarize
    schedule = evaluation_schedule(scene_configurations())
    row = dict(schedule[0], attempted=True, success=False, failure_reason='timeout')
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        summarize([row, row], schedule)
    row['attempt_id'] = 'invented'
    with pytest.raises(ValueError, match='registered|schedule'):
        summarize([row], schedule)


def test_language_pairs_measure_changed_neural_chunks_and_repeated_inside_execution():
    from ring_learning.vla_evaluate import evaluation_schedule, paired_diagnostics, scene_configurations
    schedule = evaluation_schedule(scene_configurations())
    scene = schedule[0]['configuration_id']
    rows = []
    for request in schedule:
        if request['configuration_id'] != scene:
            continue
        chunk = np.zeros((23, 4))
        if request['instruction_index'] == 1:
            chunk[:, 0] -= .06
        if request['instruction_index'] == 2:
            chunk[:, 0] += .06
        rows.append(dict(request, success=True, predicted_waypoints=chunk.tolist(),
            input_image_sha256='b'*64 if request['suite'] == 'black_image' else 'a'*64,
            trajectory=[{'action': [0., 0., 0., 1.]}]))
    result = paired_diagnostics(rows)
    assert result['language']['complete_pairs'] == 1
    assert result['language']['same_rgb_for_all_three_commands'] == 1
    assert result['language']['both_side_chunks_changed_from_inside'] == 1
    assert result['language']['mean_max_abs_waypoint_change_m']['left'] == pytest.approx(.06)
    assert result['inside_repeat']['compared'] == result['inside_repeat']['identical_trajectories'] == 1
    assert result['black_image']['compared'] == result['black_image']['different_image_hash'] == 1
