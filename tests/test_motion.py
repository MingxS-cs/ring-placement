import copy
import importlib
import json

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('ring_learning.motion')
    except ModuleNotFoundError:
        return None


def test_image_path_maps_known_diameter_and_flips_image_y():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    xy = module.image_relative_xy([[120, 180], [100, 200]], [100, 200], 200)
    np.testing.assert_allclose(xy, [[.005, .005], [0, 0]])


def test_path_has_arc_progress_and_real_curvature_residual():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    path = module.build_path([[.04, 0], [.02, .01], [0, 0]], [10, 12, 14])
    np.testing.assert_allclose(path['progress'], [0, .5, 1])
    np.testing.assert_allclose(path['residual_xy_m'], [[0, 0], [0, .01], [0, 0]])
    assert path['arc_length_m'] > path['straight_distance_m']


def test_short_interpolation_keeps_provenance_and_never_bridges_long_gap():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    samples = [
        {'source_frame': 0, 'center_px': [0, 0], 'kind': 'observed'},
        {'source_frame': 2, 'center_px': None, 'kind': 'missing'},
        {'source_frame': 4, 'center_px': [4, 0], 'kind': 'observed'},
        {'source_frame': 20, 'center_px': None, 'kind': 'missing'},
        {'source_frame': 40, 'center_px': [40, 0], 'kind': 'observed'},
    ]
    result = module.fill_short_gaps(samples, max_gap_frames=6)
    assert result[1]['kind'] == 'short_gap_interpolated'
    assert result[1]['center_px'] == [2, 0]
    assert result[1]['interpolation_bracket_frames'] == [0, 4]
    assert result[3]['center_px'] is None
    assert samples[1]['center_px'] is None


@pytest.mark.parametrize('diameter', [0, -1, float('nan')])
def test_nonmetric_mapping_rejects_invalid_scale(diameter):
    module = api()
    assert module is not None, 'motion extraction module must exist'
    with pytest.raises(ValueError):
        module.image_relative_xy([[0, 0]], [0, 0], diameter)


@pytest.mark.parametrize('frames', [[1, 1, 2], [3, 2, 1]])
def test_path_rejects_duplicate_or_reversed_source_frames(frames):
    module = api()
    assert module is not None, 'motion extraction module must exist'
    with pytest.raises(ValueError):
        module.build_path([[.04, 0], [.02, .01], [0, 0]], frames)


def test_loading_checks_episode_boundaries_and_no_duplicate_lineage(tmp_path):
    module = api()
    assert module is not None, 'motion extraction module must exist'
    record = {'episode_id': 'E01', 'source_frame_start': 10,
              'source_frame_terminal': 14, 'status': 'accepted',
              'samples': [{'source_frame': 10}, {'source_frame': 12}, {'source_frame': 14}],
              'path': module.build_path([[.04, 0], [.02, .01], [0, 0]], [10, 12, 14])}
    payload = {'schema_version': 1, 'source': {'sha256': 'a' * 64}, 'episodes': [record]}
    path = tmp_path / 'motion.json'
    path.write_text(json.dumps(payload))
    assert set(module.load_motion(path)) == {'E01'}
    payload['episodes'][0]['samples'][1]['source_frame'] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='boundary'):
        module.load_motion(path)
    payload['episodes'] = [record, copy.deepcopy(record)]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.load_motion(path)


def test_path_rejects_nonfinite_or_stationary_motion():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    for points in ([[0, 0], [float('nan'), 0]], [[0, 0], [0, 0]]):
        with pytest.raises(ValueError):
            module.build_path(points, [1, 2])


def test_reviewed_shadow_is_removed_and_fill_remains_explicit():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    samples = [{'source_frame': f, 'center_px': [float(f), 0],
                'kind': 'observed', 'confidence': .8} for f in [10, 12, 14]]
    result = module.apply_sample_review(samples, {'reject_source_frames': [12]})
    assert result[1]['kind'] == 'short_gap_interpolated'
    assert result[1]['confidence'] == 0
    assert result[1]['rejected_center_px'] == [12, 0]
    assert result[1]['manual_review_action'] == 'reject optical contour association'
    assert samples[1]['kind'] == 'observed'


def test_dark_ring_detection_ignores_displaced_brighter_shadow():
    module = api()
    assert module is not None, 'motion extraction module must exist'
    import cv2
    image = np.full((400, 400, 3), 180, np.uint8)
    cv2.circle(image, (200, 180), 45, (20, 20, 20), 8)
    cv2.circle(image, (200, 280), 45, (115, 115, 115), 8)
    detected = module.detect_moving_ring(image, [200, 175], 200, 60)
    np.testing.assert_allclose(detected['center_px'], [200, 180], atol=1)


def test_rejecting_shadow_rebuilds_neighboring_interpolation_from_true_observations():
    module = api()
    samples = [
        {'source_frame': 0, 'center_px': [0, 0], 'kind': 'observed'},
        {'source_frame': 2, 'center_px': [2, 10], 'kind': 'short_gap_interpolated'},
        {'source_frame': 4, 'center_px': [4, 20], 'kind': 'observed'},
        {'source_frame': 6, 'center_px': [6, 10], 'kind': 'short_gap_interpolated'},
        {'source_frame': 8, 'center_px': [8, 0], 'kind': 'observed'},
    ]
    result = module.apply_sample_review(samples, {'reject_source_frames': [4]})
    for index in [1, 2, 3]:
        assert result[index]['center_px'] == [2*index, 0]
        assert result[index]['interpolation_bracket_frames'] == [0, 8]
