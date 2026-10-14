import json
import hashlib

import numpy as np
import pytest
import torch


def biased_model():
    from ring_learning.video_policy import ShapeNet
    model = ShapeNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias[:] = torch.tensor([0.0, 0.3])
    return model


def test_learned_waypoint_is_anchored_at_both_endpoints():
    model = biased_model()
    start, goal = np.array([0.35, -0.08]), np.array([0.45, 0.02])
    np.testing.assert_array_equal(model.waypoint(start, goal, 0), start)
    np.testing.assert_array_equal(model.waypoint(start, goal, 1), goal)
    assert np.linalg.norm(model.waypoint(start, goal, .5) - (start+goal)/2) > .005


def test_video_waypoint_co_rotates_and_translates_with_scene():
    model = biased_model()
    start, goal = np.array([0.35, -.08]), np.array([.45, .02])
    angle = .9
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.08, -.03])
    expected = rotation @ model.waypoint(start, goal, .35) + shift
    actual = model.waypoint(rotation @ start+shift, rotation @ goal+shift, .35)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_large_network_outputs_are_physically_capped_without_nan():
    model = biased_model()
    with torch.no_grad():
        model.net[-1].bias[:] = torch.tensor([1e6, 1e6])
    start, goal = np.array([0., 0.]), np.array([.2, 0.])
    residual = model.waypoint(start, goal, .5) - (start+goal)/2
    assert np.isfinite(residual).all()
    assert np.linalg.norm(residual) <= model.max_residual_m + 1e-9
    np.testing.assert_array_equal(model.waypoint(start, start, .5), start)


@pytest.mark.parametrize('start,goal,progress', [([np.nan, 0], [0, 0], .5),
    ([0, 0], [0, np.inf], .5), ([0, 0], [1, 1], np.nan),
    ([0, 0], [1, 1], -1), ([0, 0, 0], [1, 1], .5)])
def test_invalid_waypoint_inputs_fail_explicitly(start, goal, progress):
    with pytest.raises(ValueError):
        biased_model().waypoint(start, goal, progress)


class TransferEnv:
    phase = 4
    steps = 1
    command_xyz = np.array([.40, -.03, .16])

    def phase_target(self):
        return np.array([.50, .02, .164])


def test_transfer_action_uses_network_and_caps_speed():
    from ring_learning.video_policy import VideoController
    learned = VideoController(biased_model())
    straight = VideoController(biased_model(), geometry=True)
    env = TransferEnv()
    learned_action, straight_action = learned(env, None), straight(env, None)
    assert learned.learned_steps == 1 and learned.planned_steps == 0
    assert straight.learned_steps == 0 and straight.planned_steps == 1
    assert not np.allclose(learned_action[:2], straight_action[:2], atol=1e-7)
    assert np.linalg.norm(learned_action[:3]) * .003 <= .002 + 1e-8
    assert learned_action[3] == -1
    np.testing.assert_array_equal(learned.start_xy, env.command_xyz[:2])


def test_controller_restarts_progress_for_a_new_episode():
    from ring_learning.video_policy import VideoController
    controller = VideoController(biased_model())
    env = TransferEnv()
    controller(env, None)
    first_progress = controller.progress
    env.steps = 2
    controller(env, None)
    assert controller.progress > first_progress
    env.steps = 1
    env.command_xyz = np.array([.43, .01, .16])
    controller(env, None)
    np.testing.assert_array_equal(controller.start_xy, env.command_xyz[:2])
    assert controller.progress < .05


def test_motion_source_split_rejects_unknown_and_nonfinite_sources():
    from ring_learning.video_train import prepare_samples
    episodes = [{'episode_id': f'E{i:02d}', 'scene_group': 1 if i < 9 else 2}
                for i in range(1, 12)]
    records = {e['episode_id']: {'episode_id': e['episode_id'], 'path': {
        'progress': [0., .5, 1.], 'relative_xy_m': [[.1, 0.], [.05, .01], [0., 0.]],
        'sample_source_frames': [i*10 for i in range(3)], 'sample_kind': ['observed']*3}}
        for e in episodes}
    _, membership = prepare_samples(records, episodes)
    assert not (set(membership['train']) & set(membership['val']))
    assert not (set(membership['train']) & set(membership['test']))
    records['unknown'] = records['E01']
    with pytest.raises(ValueError, match='unknown'):
        prepare_samples(records, episodes)
    del records['unknown']
    records['E01']['path']['relative_xy_m'][1][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        prepare_samples(records, episodes)


def test_rejected_track_is_not_consumed_and_weights_respect_interpolation():
    from ring_learning.video_train import prepare_samples
    episodes = [{'episode_id': f'E{i:02d}', 'scene_group': 1 if i < 9 else 2}
                for i in range(1, 12)]
    records = {e['episode_id']: {'episode_id': e['episode_id'], 'status': 'accepted', 'path': {
        'progress': [0., .5, 1.], 'relative_xy_m': [[.1, 0.], [.05, .01], [0., 0.]],
        'sample_source_frames': [0, 5, 10], 'sample_kind': ['observed', 'interpolated', 'observed'],
        'sample_weights': [1., .25, 1.]}} for e in episodes}
    records['E09'] = {'episode_id': 'E09', 'status': 'rejected'}
    prepared, _ = prepare_samples(records, episodes)
    first = prepared['train'][0]
    # Endpoint arc cell is half the middle cell: (0.25*1)/(0.5*0.25)=2.
    assert first['weights'][0] == 2*first['weights'][1]
    assert not any(r['source'] == 'E09' for group in prepared.values() for r in group)
    records['E01']['path']['sample_weights'][1] = -1
    with pytest.raises(ValueError, match='weights'):
        prepare_samples(records, episodes)


def test_checkpoint_preserves_network_and_rejects_wrong_environment(tmp_path):
    from ring_learning.integrity import environment_signature
    from ring_learning.video_policy import save_video_policy, load_video_policy
    model = biased_model()
    path = tmp_path/'policy.pt'
    save_video_policy(path, model, {'environment': environment_signature()})
    restored, _ = load_video_policy(path)
    np.testing.assert_array_equal(restored.waypoint([0., 0.], [.1, 0.], .3),
                                  model.waypoint([0., 0.], [.1, 0.], .3))
    wrong = environment_signature()
    wrong['sim_sha256'] = '0'*64
    save_video_policy(path, model, {'environment': wrong})
    with pytest.raises(ValueError, match='environment'):
        load_video_policy(path)


def test_training_checkpoint_is_unchanged_when_heldout_motion_changes(tmp_path):
    from ring_learning.dataset import split_episodes
    from ring_learning.motion import build_path
    from ring_learning.video_policy import load_video_policy
    from ring_learning.video_train import train_motion
    episodes = [{'episode_id': f'E{i:02d}', 'scene_group': 1 if i < 9 else 2}
                for i in range(1, 12)]
    splits = split_episodes(episodes)
    heldout = {e['episode_id'] for group in ('test', 'ood_test') for e in splits[group]}
    real = tmp_path/'real.json'
    real.write_text(json.dumps({'episodes': episodes, 'source': {'sha256': 'a'*64}}), encoding='utf-8')
    payload = {'schema_version': 1, 'source': {'sha256': 'a'*64}, 'episodes': [],
               'endpoint_manifest_sha256': hashlib.sha256(real.read_bytes()).hexdigest()}
    for ep in episodes:
        payload['episodes'].append({'episode_id': ep['episode_id'], 'status': 'accepted',
            'source_frame_start': 0, 'source_frame_terminal': 10,
            'samples': [{'source_frame': frame} for frame in (0, 5, 10)],
            'path': build_path([[.1, 0.], [.05, .003], [0., 0.]], [0, 5, 10])})
    motion = tmp_path/'motion.json'
    motion.write_text(json.dumps(payload), encoding='utf-8')
    train_motion(motion, real, tmp_path/'first', epochs=8)
    for ep in payload['episodes']:
        if ep['episode_id'] in heldout:
            ep['path'] = build_path([[.1, 0.], [.05, -.012], [0., 0.]], [0, 5, 10])
    motion.write_text(json.dumps(payload), encoding='utf-8')
    train_motion(motion, real, tmp_path/'second', epochs=8)
    first, first_meta = load_video_policy(tmp_path/'first'/'policy.pt')
    second, second_meta = load_video_policy(tmp_path/'second'/'policy.pt')
    assert first_meta['best_epoch'] == second_meta['best_epoch']
    assert first_meta['best_validation_loss'] == second_meta['best_validation_loss']
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    assert not (set(first_meta['training_sources']) & heldout)


def test_arc_quadrature_avoids_bias_from_dense_idle_samples():
    from ring_learning.video_train import arc_cell_weights
    uniform = np.array([0., .5, 1.])
    clustered = np.array([0., .01, .02, .04, .05, .5, 1.])
    a, b = arc_cell_weights(uniform), arc_cell_weights(clustered)
    np.testing.assert_allclose(a, [.25, .5, .25])
    np.testing.assert_allclose(a@uniform, b@clustered, atol=1e-12)
    assert b[:5].sum() < .3
    duplicate_start = arc_cell_weights([0., 0., .5, 1.])
    assert duplicate_start[0] == 0
    assert duplicate_start.sum() == 1


def test_contour_confidence_is_not_treated_as_calibrated_loss_weight():
    from ring_learning.video_train import prepare_samples
    episodes = [{'episode_id': f'E{i:02d}', 'scene_group': 1 if i < 9 else 2}
                for i in range(1, 12)]
    records = {e['episode_id']: {'episode_id': e['episode_id'], 'status': 'accepted', 'path': {
        'progress': [0., .5, 1.], 'relative_xy_m': [[.1, 0.], [.05, .01], [0., 0.]],
        'sample_kind': ['observed', 'interpolated', 'observed'],
        'sample_weights': [.4, .25, .9]}} for e in episodes}
    prepared, _ = prepare_samples(records, episodes)
    np.testing.assert_allclose(prepared['train'][0]['weights'], [.4, .2, .4], atol=1e-7)


@pytest.mark.parametrize('mismatch', ['source', 'endpoint_manifest'])
def test_training_rejects_mismatched_video_and_endpoint_manifest(tmp_path, mismatch):
    from ring_learning.video_train import train_motion
    real = tmp_path/'real.json'
    real.write_text(json.dumps({'episodes': [], 'source': {'sha256': 'a'*64}}), encoding='utf-8')
    payload = {'schema_version': 1, 'source': {'sha256': 'a'*64}, 'episodes': [],
               'endpoint_manifest_sha256': hashlib.sha256(real.read_bytes()).hexdigest()}
    if mismatch == 'source':
        payload['source']['sha256'] = 'b'*64
    else:
        payload['endpoint_manifest_sha256'] = '0'*64
    motion = tmp_path/'motion.json'
    motion.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        train_motion(motion, real, tmp_path/'output', epochs=1)
    assert not (tmp_path/'output'/'policy.pt').exists()
