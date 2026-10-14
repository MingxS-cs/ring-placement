import hashlib
import json

import mujoco
import numpy as np
import pytest

from ring_learning.sim import RingEnv


def _module():
    # Import in each test so the pre-implementation run reports feature failures.
    from ring_learning import vla_env
    return vla_env


def test_policy_observation_contains_only_own_proprioception():
    api = _module()
    env = api.VLAEnv()
    try:
        first = env.reset((.37, -.12), (.49, .05), seed=1)
        second = env.reset((.60, .16), (.42, -.07), seed=2)
        assert first.shape == second.shape == (7,)
        np.testing.assert_allclose(first, second, atol=1e-7)
        np.testing.assert_allclose(second[:3], env.tool_xyz, atol=1e-7)
        np.testing.assert_allclose(second[3:6], env.command_xyz-env.tool_xyz, atol=1e-7)
        assert second[6] == pytest.approx(env.finger_opening)
        env.phase = 5
        np.testing.assert_array_equal(env.observe(), second)
    finally:
        env.close()


def test_vla_step_never_requests_an_oracle_phase_target():
    env = _module().VLAEnv(max_steps=3)
    try:
        env.phase = 3
        env.phase_steps = 19
        def forbidden():
            raise AssertionError('Oracle phase target reached inference physics')
        env.phase_target = forbidden
        for index in range(3):
            observation, done, info = env.step([0., 0., 0., 1.])
            assert observation.shape == (7,)
            assert done is (index == 2)
            assert info['failure_reason'] != 'failed_lift'
        assert info['failure_reason'] == 'timeout'
    finally:
        env.close()


def test_action_execution_matches_unchanged_base_physics():
    env = _module().VLAEnv()
    base = RingEnv()
    try:
        env.reset((.4, -.09), (.49, .04), seed=5)
        base.reset((.4, -.09), (.49, .04), seed=5)
        rng = np.random.default_rng(123)
        for action in rng.uniform(-.5, .5, (6, 4)):
            env.step(action)
            base.step(action)
        np.testing.assert_array_equal(env.command_xyz, base.command_xyz)
        np.testing.assert_array_equal(env.data.qpos, base.data.qpos)
        np.testing.assert_array_equal(env.data.qvel, base.data.qvel)
        np.testing.assert_array_equal(env.data.ctrl, base.data.ctrl)
        assert env.data.time == base.data.time
    finally:
        env.close()
        base.close()


def test_fixed_camera_projection_round_trip_and_extent():
    api = _module()
    env = api.VLAEnv()
    try:
        xy = np.array([[.47, 0.], [.19, .28], [.75, -.28], [.32, -.23], [.62, .23]])
        projected = env.project_xy(xy)
        np.testing.assert_allclose(projected[:3], [[.5, .5], [0., 0.], [1., 1.]], atol=1e-12)
        np.testing.assert_allclose(env.pixel_to_world(projected), xy, atol=1e-12)
        assert np.all((projected[3:] > 0) & (projected[3:] < 1))
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'tabletop')
        assert env.model.cam_fovy[camera_id] == pytest.approx(.56)
    finally:
        env.close()


@pytest.mark.parametrize('ring,bearing', [((.37, -.11), (.51, .075)), ((.585, .19), (.415, -.085))])
def test_actual_render_mask_centroids_match_fixed_calibration(ring, bearing):
    api = _module()
    env = api.VLAEnv()
    try:
        env.reset(ring, bearing, seed=8)
        rgb = env.render_policy()
        masks = env.render_masks()
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        assert masks.shape == (2, 256, 256) and masks.dtype == np.bool_
        for mask, xy in zip(masks, (ring, bearing)):
            rows, columns = np.nonzero(mask)
            assert len(rows) >= 20
            center = (np.array([columns.mean(), rows.mean()])+.5)/api.IMAGE_SIZE
            error_pixels = np.linalg.norm(center-env.project_xy(xy))*api.IMAGE_SIZE
            assert error_pixels < .35
        # Training-only segmentation toggles are fully restored before RGB.
        np.testing.assert_array_equal(env.render_policy(), rgb)
    finally:
        env.close()


def test_policy_view_hides_robot_but_wide_demo_keeps_it_visible():
    env = _module().VLAEnv()
    try:
        policy = env.render_policy()
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1
        env._policy_renderer.update_scene(env.data, camera='tabletop', scene_option=option)
        with_robot = env._policy_renderer.render().copy()
        assert np.count_nonzero(np.any(policy != with_robot, axis=2)) > 100
        wide = env.render()
        assert wide.shape == (480, 640, 3)
        ids = [int(env.renderer.scene.geoms[i].objid)
               for i in range(env.renderer.scene.ngeom)
               if int(env.renderer.scene.geoms[i].objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)]
        assert any(env.model.geom_group[i] == 5 for i in ids)
    finally:
        env.close()


def test_camera_signature_is_static_and_canonical():
    api = _module()
    first = api.camera_signature()
    second = api.camera_signature()
    assert first == second
    encoded = json.dumps(first['specification'], sort_keys=True, separators=(',', ':')).encode()
    assert first['sha256'] == hashlib.sha256(encoded).hexdigest()
    assert first['specification']['robot_geometry_hidden'] is True
    assert first['specification']['crop_uses_object_coordinates'] is False


def test_explicit_finish_is_separate_from_object_state():
    env = _module().VLAEnv()
    try:
        env.reset()
        assert not env.finish()
        _, done, info = env.step([0, 0, 0, 1])
        assert done
        assert info['failure_reason'] == 'controller_finished_without_success'
        env.reset()
        _, done, _ = env.step([0, 0, 0, 1])
        assert not done
    finally:
        env.close()


def test_runtime_actions_reject_nonfinite_and_bad_shapes():
    env = _module().VLAEnv()
    try:
        for action in ([0, 0, 1], [0, np.nan, 0, 1]):
            with pytest.raises(ValueError, match='finite length4'):
                env.step(action)
        before = env.data.qpos.copy()
        env.step([0, 0, 0, 1])
        assert not np.array_equal(before, env.data.qpos)
    finally:
        env.close()


@pytest.mark.parametrize('instruction,offset', [(0, 0.), (1, -.06), (2, .06)])
def test_instruction_scoring_requires_released_settled_correct_target(instruction, offset):
    api = _module()
    env = api.VLAEnv()
    try:
        bearing = np.array([.48, .02])
        env.reset(bearing+[offset, 0.], bearing, seed=9)
        assert not api.score_target(env, instruction)
        for _ in range(12):
            _, done, _ = env.step([0, 0, 0, 1.])
            assert not done
        assert api.score_target(env, instruction)
        assert all(not api.score_target(env, other) for other in range(3) if other != instruction)
        env.data.qvel[env.ring_dof_adr] = .02
        assert not api.score_target(env, instruction)
        env.data.qvel[env.ring_dof_adr] = 0
        # Inside has 7.5 mm radial clearance; left/right allow only 3 mm.
        env.data.qpos[env.ring_qpos_adr] += .009 if instruction == 0 else .004
        mujoco.mj_forward(env.model, env.data)
        assert not api.score_target(env, instruction)
    finally:
        env.close()


def test_scoring_accepts_declared_text_commands_and_rejects_unknown():
    api = _module()
    env = api.VLAEnv()
    try:
        for index, command in enumerate(api.COMMANDS):
            assert api.score_target(env, command) is api.score_target(env, index)
        with pytest.raises(ValueError, match='instruction'):
            api.score_target(env, 'delete table')
    finally:
        env.close()
