import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class Curve:
    def waypoint(self, start, goal, progress, reference_angle=0):
        return (1-progress)*np.asarray(start)+progress*np.asarray(goal)+[0, .015*4*progress*(1-progress)]


def environment():
    env = SimpleNamespace(phase=4, steps=0, command_xyz=np.array([.4, 0., .16]),
                          tool_xyz=np.array([.4, 0., .16]))
    env.phase_target = lambda: np.array([.5, 0., .16])
    return env


def test_curve_execution_limits_displacement_and_reaches_exact_endpoint():
    module = importlib.import_module('ring_learning.generalization_control')
    env = environment()
    controller = module.GeneralizationController(Curve())
    positions = []
    for _ in range(100):
        old = env.command_xyz.copy()
        action = controller(env, None)
        env.command_xyz += module.ACTION_SCALE*action[:3]
        env.tool_xyz = env.command_xyz.copy()
        env.steps += 1
        assert np.linalg.norm(env.command_xyz-old) <= .002000001
        assert np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 1
        positions.append(env.command_xyz.copy())
    assert env.command_xyz == pytest.approx([.5, 0., .16], abs=1e-7)
    assert controller.progress == 1.
    assert max(p[1] for p in positions) > .014
    assert controller.curve_length_m > .1


def test_tool_lag_pauses_reference_advance_without_state_injection():
    module = importlib.import_module('ring_learning.generalization_control')
    env = environment()
    controller = module.GeneralizationController(Curve())
    controller(env, None)
    before = controller.arc_cursor_m
    env.steps += 1
    env.tool_xyz = env.command_xyz-[.02, 0, 0]
    saved_tool = env.tool_xyz.copy()
    controller(env, None)
    assert controller.arc_cursor_m == before
    assert env.tool_xyz == pytest.approx(saved_tool)


def test_geometry_ablation_does_not_call_model():
    module = importlib.import_module('ring_learning.generalization_control')
    class MustNotBeCalled:
        def waypoint(self, *args, **kwargs):
            raise AssertionError('Straight reference must not evaluate learned model')
    controller = module.GeneralizationController(MustNotBeCalled(), geometry=True)
    action = controller(environment(), None)
    assert action[1] == 0
    assert controller.learned_reference_evaluations == 0


def test_reset_reuses_controller_without_old_reference_or_counters():
    module = importlib.import_module('ring_learning.generalization_control')
    env = environment()
    controller = module.GeneralizationController(Curve())
    controller(env)
    env.steps = 5
    controller(env)
    env.steps = 0
    env.command_xyz = np.array([.42, .01, .16])
    env.tool_xyz = env.command_xyz.copy()
    controller(env)
    assert controller.start_xy == pytest.approx([.42, .01])
    assert controller.transfer_steps == 1
    assert controller.learned_reference_evaluations == 63


def test_nonfinite_reference_is_rejected_before_action():
    module = importlib.import_module('ring_learning.generalization_control')
    class Invalid:
        def waypoint(self, *args, **kwargs):
            return [np.nan, 0.]
    with pytest.raises(ValueError, match='finite planar'):
        module.GeneralizationController(Invalid())(environment())


def test_nontransfer_uses_existing_geometry_and_never_evaluates_model():
    module = importlib.import_module('ring_learning.generalization_control')
    env = environment()
    env.phase = 3
    controller = module.GeneralizationController(None)
    action = controller(env)
    assert action == pytest.approx([0., 0., 0., -1.])
    assert controller.planned_steps == 1
    assert controller.learned_reference_evaluations == 0


def test_curve_cannot_enter_lowering_on_an_early_pass_through_goal():
    module = importlib.import_module('ring_learning.generalization_control')
    from ring_learning.generalization_policy import DirectionRidgePolicy
    model = DirectionRidgePolicy([[.025, 0.], [0., 0.], [0., 0.]], [0., 0.], [1., 1.])
    env = module.GeneralizationEnv()
    try:
        observation = env.reset([.400, 0.], [.445, 0.], 0)
        controller = module.GeneralizationController(model)
        blocks, saw_transfer = 0, False
        for _ in range(env.max_steps):
            old_phase = env.phase
            observation, done, info = env.step(controller(env, observation))
            blocks += info['reference_completion_guard_blocked']
            if old_phase == 4:
                saw_transfer = True
                if env.phase == 5:
                    assert controller.progress == pytest.approx(1.)
                    break
            if done:
                pytest.fail('Synthetic reference did not complete physical transfer')
        assert saw_transfer and blocks > 0
        assert env.phase == 5
    finally:
        env.close()
