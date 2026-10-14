import numpy as np
import pytest


def test_transfer_window_includes_before_and_after_action_without_truth_filtering():
    from ring_learning.motion_ablation_analysis import transfer_path
    row = {'controller_waypoint_indices': [3, 4, 5, 19, 20],
           'trajectory': [{'ring_xyz_m': [i, i*i, 0.1]} for i in range(6)]}
    np.testing.assert_array_equal(transfer_path(row), [[1, 1], [2, 4], [3, 9], [4, 16]])


def test_arc_resampling_compares_shape_despite_different_speed_and_duplicate_points():
    from ring_learning.motion_ablation_analysis import sample_by_arc_length
    first = np.array([[0, 0], [0.5, 0], [1, 0]])
    second = np.array([[0, 0], [0, 0], [0.1, 0], [0.9, 0], [1, 0]])
    np.testing.assert_allclose(sample_by_arc_length(first), sample_by_arc_length(second), atol=1e-15)


def test_missing_transfer_stays_unavailable_and_stationary_path_stays_degenerate():
    from ring_learning.motion_ablation_analysis import sample_by_arc_length, transfer_path
    assert transfer_path({'controller_waypoint_indices': [], 'trajectory': []}).shape == (0, 2)
    assert sample_by_arc_length(np.empty((0, 2))) is None
    np.testing.assert_array_equal(sample_by_arc_length(np.array([[1, 2], [1, 2]])), np.tile([1, 2], (100, 1)))


def test_path_stats_measure_shape_instead_of_sample_count():
    from ring_learning.motion_ablation_analysis import path_stats
    curved = path_stats(np.array([[0, 0], [0.5, 0.25], [1, 0]]))
    assert curved['path_length_mm'] == pytest.approx(np.sqrt(1.25)*1000)
    assert curved['max_lateral_mm'] == 250
    assert curved['arc_to_chord'] == pytest.approx(np.sqrt(1.25))
    stationary = path_stats(np.array([[1, 2], [1, 2]]))
    assert stationary['path_length_mm'] == 0 and stationary['arc_to_chord'] is None
    assert path_stats(np.array([[0., 0.], [1e-10, 0.]]))['arc_to_chord'] is None


def paired_report():
    chunks = np.zeros((23, 4))
    base = dict(configuration_id='one', input_image_sha256='same', predicted_waypoints=chunks.tolist(),
                trajectory=[], controller_waypoint_indices=[], success=False, controller_finished=False,
                ring_xy_m=[.45, .05], bearing_xy_m=[.47, 0.], reset_seed=7,
                instruction='put the ring inside the bearing',
                policy_forbidden_access_attempts=0, steps=0)
    second = dict(base, suite='straight_transfer')
    modified = chunks.copy(); modified[4:19, :2] = 0.01
    return {'rollouts': [dict(base, suite='video_motion', predicted_waypoints=modified.tolist()), second]}


def test_pair_audit_retains_failures_and_checks_the_only_allowed_action_change():
    from ring_learning.motion_ablation_analysis import audit_pairs
    result = audit_pairs(paired_report(), expected_pairs=1)
    assert result['attempts'] == 2
    assert result['same_images'] == result['common_actions_equal'] == 1
    assert result['successes'] == {'video_motion': 0, 'straight_transfer': 0}


def test_pair_audit_rejects_missing_records_changed_images_or_critical_actions():
    from ring_learning.motion_ablation_analysis import audit_pairs
    report = paired_report(); report['rollouts'].pop()
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)
    report = paired_report(); report['rollouts'][1]['input_image_sha256'] = 'different'
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)
    report = paired_report(); report['rollouts'][1]['predicted_waypoints'][0][0] = 0.01
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)


def test_finished_record_needs_observation_action_and_execution_proof():
    from ring_learning.motion_ablation_analysis import audit_pairs
    report = paired_report(); report['rollouts'][0]['controller_finished'] = True
    report['rollouts'][0]['input_image_sha256'] = None
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)


def test_oracle_access_or_changed_reset_invalidates_a_matched_experiment():
    from ring_learning.motion_ablation_analysis import audit_pairs
    report = paired_report(); report['rollouts'][0]['policy_forbidden_access_attempts'] = 1
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)
    report = paired_report(); report['rollouts'][1]['reset_seed'] = 8
    with pytest.raises(ValueError): audit_pairs(report, expected_pairs=1)


def test_existing_analysis_directory_is_preserved(tmp_path):
    from ring_learning.motion_ablation_analysis import analyze
    output = tmp_path/'existing'; output.mkdir()
    (output/'keep.txt').write_text('preserve')
    with pytest.raises(FileExistsError): analyze(tmp_path/'missing.json', output)
    assert (output/'keep.txt').read_text() == 'preserve'
