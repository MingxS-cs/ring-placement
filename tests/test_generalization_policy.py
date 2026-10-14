import copy
import hashlib
import json

import numpy as np
import pytest


def direction_model(degree=1, large=False):
    from ring_learning.generalization_policy import DirectionRidgePolicy
    coefficients = np.zeros((3 * (degree + 1), 2))
    coefficients[1, 1] = 1e6 if large else .012
    coefficients[2, 0] = 1e6 if large else -.004
    return DirectionRidgePolicy(coefficients, [0., 0.], [1., 1.], degree=degree, alpha=.2)


def synthetic_payloads():
    from ring_learning.motion import build_path
    episodes = []
    records = []
    for index in range(1, 25):
        source = f'E{index:02d}'
        first = index * 100
        episodes.append(dict(episode_id=source, scene_group=1 if index < 10 else 2 if index < 20 else 3,
                             source_frame_start=first, source_frame_terminal=first+20))
        angle = index * .71
        start = np.array([np.cos(angle), np.sin(angle)]) * .08
        xy = np.array([start, start/2 + [.004, .008], [0., 0.]])
        frames = [first+2, first+8, first+14]
        path = build_path(xy, frames)
        path['sample_kind'] = ['observed', 'interpolated', 'observed']
        records.append(dict(episode_id=source, status='accepted',
                            source_frame_start=first, source_frame_terminal=first+20,
                            samples=[dict(source_frame=f) for f in frames], path=path))
    real = dict(source=dict(sha256='a'*64), episodes=episodes)
    motion = dict(schema_version=1, source=dict(sha256='a'*64), episodes=records)
    annotations = dict(schema_version=1, source_sha256='a'*64, episodes=[])
    return real, motion, annotations


def write_inputs(tmp_path, real, motion, annotations):
    real_path, motion_path, annotation_path = [tmp_path/n for n in ('real.json', 'motion.json', 'annotations.json')]
    real_path.write_text(json.dumps(real), encoding='utf-8')
    motion = copy.deepcopy(motion)
    motion['endpoint_manifest_sha256'] = hashlib.sha256(real_path.read_bytes()).hexdigest()
    motion_path.write_text(json.dumps(motion), encoding='utf-8')
    annotation_path.write_text(json.dumps(annotations), encoding='utf-8')
    return motion_path, real_path, annotation_path


def test_same_length_different_directions_can_predict_different_local_shapes():
    model = direction_model()
    x_start, x_goal = np.array([0., 0.]), np.array([.1, 0.])
    y_start, y_goal = np.array([0., 0.]), np.array([0., .1])
    x_residual = model.waypoint(x_start, x_goal, .5) - (x_start+x_goal)/2
    y_residual = model.waypoint(y_start, y_goal, .5) - (y_start+y_goal)/2
    # E02/E04 exposed this collision in the old progress+length representation.
    np.testing.assert_allclose(x_residual, [0., .012], atol=1e-12)
    np.testing.assert_allclose(y_residual, [0., -.004], atol=1e-12)
    assert model.parameter_count == 12


def test_endpoint_zero_length_and_large_predictions_remain_exact_and_capped():
    model = direction_model(large=True)
    start, goal = np.array([.3, -.08]), np.array([.4, .04])
    np.testing.assert_array_equal(model.waypoint(start, goal, 0.), start)
    np.testing.assert_array_equal(model.waypoint(start, goal, 1.), goal)
    np.testing.assert_array_equal(model.waypoint(start, start, .5), start)
    residual = model.waypoint(start, goal, .5) - (start+goal)/2
    assert np.isfinite(residual).all()
    assert np.linalg.norm(residual) <= .025 + 1e-12


def test_rigid_translation_and_rotation_with_reference_frame_are_equivariant():
    model = direction_model()
    start, goal, progress, angle = np.array([.3, -.08]), np.array([.4, .04]), .35, .9
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.07, -.02])
    expected = rotation @ model.waypoint(start, goal, progress, reference_angle=.2) + shift
    actual = model.waypoint(rotation @ start+shift, rotation @ goal+shift, progress, reference_angle=.2+angle)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(model.waypoint(start+shift, goal+shift, progress), model.waypoint(start, goal, progress)+shift, atol=1e-12)
    # Keeping the camera/task reference fixed intentionally changes style with direction.
    fixed_reference = model.waypoint(rotation @ start+shift, rotation @ goal+shift, progress, reference_angle=.2)
    assert not np.allclose(fixed_reference, expected, atol=1e-6)


@pytest.mark.parametrize('start,goal,p,angle', [([np.nan, 0], [0, 0], .5, 0),
    ([0, 0], [0, np.inf], .5, 0), ([0, 0], [1, 1], np.nan, 0),
    ([0, 0], [1, 1], -1, 0), ([0, 0, 0], [1, 1], .5, 0),
    ([0, 0], [1, 1], .5, np.inf)])
def test_invalid_waypoint_geometry_is_rejected(start, goal, p, angle):
    with pytest.raises(ValueError):
        direction_model().waypoint(start, goal, p, reference_angle=angle)


def test_manual_record_precedence_and_interpolation_weights():
    from ring_learning.generalization_train import prepare_development_samples
    from ring_learning.dataset import split_episodes
    real, motion, annotations = synthetic_payloads()
    source = split_episodes(real['episodes'])['train'][0]['episode_id']
    manual = copy.deepcopy(next(r for r in motion['episodes'] if r['episode_id'] == source))
    manual['annotation_method'] = 'manual_visual'
    manual['path']['sample_kind'] = ['manual_visual']*3
    manual['samples'] = [dict(source_frame=f, kind='manual_visual', relative_xy_m=xy)
                         for f,xy in zip(manual['path']['sample_source_frames'], manual['path']['relative_xy_m'])]
    annotations['episodes'].append(manual)
    prepared, _ = prepare_development_samples(motion, real, annotations)
    selected = next(g for g in prepared['train'] if g['source'] == source)
    assert selected['origin'] == 'manual_development_annotation'
    assert selected['sample_kind'] == ['observed']*3
    # A separately retained old record keeps its interpolated multiplier.
    old = next(g for g in prepared['train'] if g['source'] != source)
    p = old['progress']
    cells = np.r_[(p[1]-p[0])/2, (p[2:]-p[:-2])/2, (p[-1]-p[-2])/2]
    expected = cells*np.array([1., .25, 1.])
    np.testing.assert_allclose(old['weights'], expected/expected.sum())


def test_heldout_label_mutations_cannot_change_folds_or_selected_model(tmp_path):
    from ring_learning.generalization_train import train_generalization
    from ring_learning.dataset import split_episodes
    real, motion, annotations = synthetic_payloads()
    paths = write_inputs(tmp_path, real, motion, annotations)
    first, first_meta = train_generalization(*paths, tmp_path/'first')
    heldout = {e['episode_id'] for name, group in split_episodes(real['episodes']).items() if name in ('test', 'ood_test') for e in group}
    for r in motion['episodes']:
        if r['episode_id'] in heldout:
            r['path'] = {'relative_xy_m': [[float('nan'), 0]]}
            r['samples'] = None
    paths = write_inputs(tmp_path, real, motion, annotations)
    second, second_meta = train_generalization(*paths, tmp_path/'second')
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.direction_mean, second.direction_mean)
    np.testing.assert_array_equal(first.direction_std, second.direction_std)
    assert first_meta['selection'] == second_meta['selection']
    assert not set(first_meta['training_sources']) & heldout
    for candidate in first_meta['selection']['candidates']:
        for fold in candidate['folds']:
            assert fold['source'] not in fold['fit_sources']
            assert not set(fold['fit_sources']) & heldout


def test_direction_normalizer_is_refitted_only_on_fold_training_sources():
    from ring_learning.generalization_train import fit_direction_ridge
    groups = [dict(source=f'S{i}', direction=np.array(d), progress=np.array([0., .5, 1.]),
                   local_residual_m=np.zeros((3, 2)), weights=np.array([.25, .5, .25]))
              for i, d in enumerate([[1., 0.], [0., 1.], [-1., 0.]])]
    model = fit_direction_ridge(groups[:2], degree=0, alpha=.2)
    np.testing.assert_array_equal(model.direction_mean, [.5, .5])
    np.testing.assert_array_equal(model.direction_std, [.5, .5])
    other = fit_direction_ridge(groups, degree=0, alpha=.2)
    assert not np.array_equal(model.direction_mean, other.direction_mean)


@pytest.mark.parametrize('problem', ['old_coordinate_mismatch', 'manual_coordinate_missing', 'manual_pixel_mismatch'])
def test_raw_point_lineage_matches_derived_path(problem):
    from ring_learning.generalization_train import prepare_development_samples
    from ring_learning.dataset import split_episodes
    real, motion, annotations = synthetic_payloads()
    source = split_episodes(real['episodes'])['train'][0]['episode_id']
    selected = next(r for r in motion['episodes'] if r['episode_id'] == source)
    if problem == 'old_coordinate_mismatch':
        selected['samples'][1]['relative_xy_m'] = [.2, .2]
    else:
        manual = copy.deepcopy(selected)
        manual['path']['sample_kind'] = ['manual_visual']*3
        annotations['episodes'].append(manual)
        if problem == 'manual_pixel_mismatch':
            manual['bearing_center_px'] = [0., 0.]
            manual['bearing_outer_diameter_px'] = 50.
            for sample in manual['samples']:
                sample['center_px'] = [100., 100.]
    with pytest.raises(ValueError, match='coordinate|lineage|manual'):
        prepare_development_samples(motion, real, annotations)


@pytest.mark.parametrize('problem', ['source', 'annotation_source', 'conflicting_annotation_source', 'endpoint_manifest', 'nonfinite', 'bounds', 'progress', 'lineage', 'annotation_heldout'])
def test_provenance_and_selected_label_integrity_fail_before_checkpoint(tmp_path, problem):
    from ring_learning.generalization_train import train_generalization
    from ring_learning.dataset import split_episodes
    real, motion, annotations = synthetic_payloads()
    source = split_episodes(real['episodes'])['train'][0]['episode_id']
    selected = next(r for r in motion['episodes'] if r['episode_id'] == source)
    if problem == 'source': motion['source']['sha256'] = 'b'*64
    if problem == 'annotation_source': annotations['source_sha256'] = 'b'*64
    if problem == 'conflicting_annotation_source': annotations['source'] = dict(sha256='b'*64)
    if problem == 'nonfinite': selected['path']['relative_xy_m'][1][0] = float('nan')
    if problem == 'bounds': selected['path']['sample_source_frames'][0] = 0
    if problem == 'progress': selected['path']['progress'][1] = .01
    if problem == 'lineage': selected['samples'] = selected['samples'][:-1]
    if problem == 'annotation_heldout':
        heldout = split_episodes(real['episodes'])['ood_test'][0]['episode_id']
        annotations['episodes'] = [copy.deepcopy(next(r for r in motion['episodes'] if r['episode_id'] == heldout))]
    paths = write_inputs(tmp_path, real, motion, annotations)
    if problem == 'endpoint_manifest':
        saved = json.loads(paths[0].read_text())
        saved['endpoint_manifest_sha256'] = '0'*64
        paths[0].write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        train_generalization(*paths, tmp_path/'output')
    assert not (tmp_path/'output'/'policy.json').exists()


@pytest.mark.parametrize('role', [0, 1, 2])
def test_blind_artifact_barrier_covers_all_input_roles_before_any_read(tmp_path, role):
    from ring_learning.generalization_train import load_development_samples
    paths = [tmp_path/'missing_motion.json', tmp_path/'missing_real.json', tmp_path/'missing_annotation.json']
    paths[role] = tmp_path/'V4_HOLDOUT_ANNOTATIONS.JSON'
    with pytest.raises(ValueError, match='blind holdout'):
        load_development_samples(*paths)


def test_saved_policy_roundtrip_and_source_environment_binding(tmp_path):
    from ring_learning.generalization_policy import save_generalization_policy, load_generalization_policy, source_signature
    from ring_learning.integrity import environment_signature
    path = tmp_path/'policy.json'
    model = direction_model()
    metadata = dict(implementation_sources=source_signature(), environment=environment_signature())
    save_generalization_policy(path, model, metadata)
    restored, _ = load_generalization_policy(path)
    np.testing.assert_array_equal(restored.waypoint([0., 0.], [.1, .02], .4), model.waypoint([0., 0.], [.1, .02], .4))
    for key in ('implementation_sources', 'environment'):
        wrong = copy.deepcopy(metadata)
        first = next(iter(wrong[key]))
        wrong[key][first] = 'changed'
        save_generalization_policy(path, model, wrong)
        with pytest.raises(ValueError, match='source|environment'):
            load_generalization_policy(path)
