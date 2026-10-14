import numpy as np
import mujoco

from ring_learning.sim import RingEnv, PHASES, OBS_SIZE, ACTION_SIZE


def test_annular_collision_geometry_preserves_both_holes():
    env = RingEnv()
    try:
        env.reset((0.40, -0.08), (0.48, 0.02), seed=1)
        for name, inner_radius, count in [("ring", 0.010, 48), ("bearing", 0.020, 64)]:
            ids = [i for i in range(env.model.ngeom)
                   if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(name + "_")]
            assert len(ids) == count
            for i in ids:
                assert env.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                mesh = env.model.geom_dataid[i]
                start = env.model.mesh_vertadr[mesh]
                n = env.model.mesh_vertnum[mesh]
                vertices = env.model.mesh_vert[start:start+n]
                # Compiled meshes have their centroid removed: undo mesh transform.
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, env.model.mesh_quat[mesh])
                vertices = vertices @ rot.reshape(3, 3).T + env.model.mesh_pos[mesh]
                assert np.linalg.norm(vertices[:, :2], axis=1).min() >= inner_radius - 1e-7
        assert env.model.body_jntnum[env.ring_body_id] == 1
        assert env.model.jnt_type[env.ring_joint_id] == mujoco.mjtJoint.mjJNT_FREE
        assert not any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    finally:
        env.close()


def test_strict_success_needs_released_settled_containment():
    env = RingEnv()
    try:
        obs = env.reset((0.48, 0.02), (0.48, 0.02), seed=3)
        assert obs.shape == (OBS_SIZE,)
        assert ACTION_SIZE == 4
        assert not env.success()
        for _ in range(12):
            env.step(np.array([0, 0, 0, 1.0]))
        assert env.success()
        env.data.qpos[env.ring_qpos_adr] += 0.009
        mujoco.mj_forward(env.model, env.data)
        assert not env.success()
        env.data.qpos[env.ring_qpos_adr] -= 0.009
        env.data.qvel[env.ring_dof_adr] = 0.02
        assert not env.success()
        env.data.qvel[env.ring_dof_adr] = 0
        env.data.qpos[env.ring_qpos_adr+3:env.ring_qpos_adr+7] = [np.cos(np.pi/8), np.sin(np.pi/8), 0, 0]
        mujoco.mj_forward(env.model, env.data)
        assert not env.success()
        env.data.qpos[env.ring_qpos_adr+3:env.ring_qpos_adr+7] = [1, 0, 0, 0]
        env.data.qpos[env.ring_qpos_adr + 2] += 0.020
        mujoco.mj_forward(env.model, env.data)
        assert not env.success()
    finally:
        env.close()


def test_reset_uses_fixed_home_independent_of_ring_position():
    env = RingEnv()
    try:
        env.reset((0.39, -0.07), (0.48, 0.02), seed=4)
        tool = env.tool_xyz
        env.reset((0.55, 0.08), (0.47, -0.02), seed=5)
        assert np.allclose(env.tool_xyz, tool, atol=1e-6)
        assert np.allclose(env.command_xyz, [0.45, -0.02, 0.25])
    finally:
        env.close()


def test_teacher_physically_lifts_and_releases_ring():
    from ring_learning.teacher import rollout
    env = RingEnv()
    try:
        env.reset((0.39, -0.07), (0.48, 0.02), seed=4)
        result = rollout(env, max_steps=450)
        assert result["success"], result["info"]
        assert result["max_ring_height"] > env.table_z + 0.035
        assert result["finger_contact_steps"] > 5
        assert result["actions"].shape[1] == 4
        assert set(result["phases"]) == set(range(len(PHASES)))
    finally:
        env.close()


def test_lift_completion_uses_height_and_contact_not_cached_initial_xy():
    from ring_learning.teacher import teacher
    env = RingEnv()
    try:
        env.reset((0.39, -0.07), (0.48, 0.02), seed=8)
        for _ in range(250):
            if env.phase == 3:
                break
            env.step(teacher(env))
        assert env.phase == 3
        assert env.ring_finger_contacts() > 0
        # The prior grasp-point cache is not a necessary spatial waypoint for
        # lifting: a held object may shift laterally and still be lifted safely.
        env.grasp_xyz[:2] += [.010, .010]
        for _ in range(50):
            delta_z = np.clip((env.phase_target()[2] - env.command_xyz[2]) / .003, -1, 1)
            env.step(np.array([0., 0., delta_z, -1.]))
            if env.phase != 3:
                break
        assert env.ring_xyz[2] > env.table_z + .025
        assert env.phase == 4
    finally:
        env.close()
