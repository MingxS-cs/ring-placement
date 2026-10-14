"""Execute a learned planar reference through the unchanged physical servo.

Only XY during transfer uses the model. Other stages, height and fingers use
the existing geometric controller. A reference cursor follows curve arc length
and waits for the physical tool when its own setpoint is too far ahead.
"""
from __future__ import annotations

import numpy as np

from .sim import ACTION_SCALE, RingEnv
from .teacher import teacher


class GeneralizationEnv(RingEnv):
    """Add reference completion to the transfer-stage guard only.

    The base step completes physics before its stage transition. Veto only an
    early transfer-to-lower transition; no pose, velocity, actuator, command or
    contact state is changed. Keeping the original simulator source preserves
    the frozen V3 model's physics binding.
    """
    def reset(self, *args, **kwargs):
        self.transfer_reference_complete = True
        return super().reset(*args, **kwargs)

    def set_transfer_reference_complete(self, complete):
        self.transfer_reference_complete = bool(complete)

    def step(self, action):
        old_phase, old_phase_steps = self.phase, self.phase_steps
        observation, done, info = super().step(action)
        blocked = (old_phase == 4 and self.phase == 5
                   and not self.transfer_reference_complete)
        if blocked:
            self.phase, self.phase_steps = 4, old_phase_steps+1
            if self.failure_reason == 'timeout_lower':
                self.failure_reason = 'timeout_transfer'
            observation = self.observe()
            info.update(phase=self.phase_name, phase_index=self.phase,
                        failure_reason=self.failure_reason)
        info['reference_completion_guard_blocked'] = blocked
        return observation, done, info


class GeneralizationController:
    def __init__(self, model, *, geometry=False, max_step_m=.002,
                 reference_angle=0., lag_limit_m=.004):
        if (not np.isfinite([max_step_m, reference_angle, lag_limit_m]).all()
                or not 0 < max_step_m <= ACTION_SCALE or lag_limit_m <= 0):
            raise ValueError('Reference speed and lag bounds must be finite and positive')
        self.model, self.geometry = model, bool(geometry)
        self.max_step_m, self.lag_limit_m = float(max_step_m), float(lag_limit_m)
        self.reference_angle = float(reference_angle)
        self._last_step, self._in_transfer = -1, False
        self._reset()

    def _reset(self):
        self.progress = self.arc_cursor_m = self.curve_length_m = 0.
        self.learned_reference_evaluations = self.transfer_steps = 0
        self.planned_steps = self.lag_pause_steps = 0
        self.start_xy = self.goal_xy = None
        self._in_transfer = False

    def _reference(self, env):
        self.start_xy = np.asarray(env.command_xyz[:2], float).copy()
        self.goal_xy = np.asarray(env.phase_target()[:2], float).copy()
        self._progress_grid = np.linspace(0., 1., 65)
        points = []
        for p in self._progress_grid:
            if self.geometry or p in (0., 1.):
                xy = (1-p)*self.start_xy+p*self.goal_xy
            else:
                xy = self.model.waypoint(self.start_xy, self.goal_xy, float(p),
                                         reference_angle=self.reference_angle)
                self.learned_reference_evaluations += 1
            points.append(xy)
        self._curve = np.asarray(points, float)
        if self._curve.shape != (65, 2) or not np.isfinite(self._curve).all():
            raise ValueError('Learned reference must be a finite planar curve')
        self._arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self._curve, axis=0), axis=1))]
        self.curve_length_m = float(self._arc[-1])
        self.arc_cursor_m = 0.
        self._in_transfer = True

    def __call__(self, env, observation=None):
        if env.steps < self._last_step:
            self._reset()
        self._last_step = env.steps
        if env.phase != 4:
            self._in_transfer = False
            self.planned_steps += 1
            return teacher(env)
        if not self._in_transfer:
            self._reference(env)
        self.transfer_steps += 1
        if self.geometry:
            self.planned_steps += 1
        lag = float(np.linalg.norm(np.asarray(env.tool_xyz)-env.command_xyz))
        if lag <= self.lag_limit_m:
            self.arc_cursor_m = min(self.curve_length_m, self.arc_cursor_m+self.max_step_m)
        else:
            self.lag_pause_steps += 1
        if self.curve_length_m <= 1e-12:
            self.progress, xy = 1., self.goal_xy
        else:
            self.progress = float(np.interp(self.arc_cursor_m, self._arc, self._progress_grid))
            xy = np.array([np.interp(self.arc_cursor_m, self._arc, self._curve[:, axis])
                           for axis in range(2)])
        if hasattr(env, 'set_transfer_reference_complete'):
            env.set_transfer_reference_complete(self.progress >= 1.-1e-12)
        target = np.r_[xy, env.phase_target()[2]]
        delta = target-np.asarray(env.command_xyz)
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step_m:
            delta *= self.max_step_m/norm
        return np.r_[delta/ACTION_SCALE, -1.].astype(np.float32)


class LegacyProfileAdapter:
    """Compare the frozen V3 model using the same V4 execution mechanism."""
    def __init__(self, model):
        self.model = model

    def waypoint(self, start_xy, goal_xy, progress, reference_angle=0.):
        if reference_angle != 0.:
            raise ValueError('Legacy comparison is defined in the fixed camera frame')
        return self.model.waypoint(start_xy, goal_xy, progress)
