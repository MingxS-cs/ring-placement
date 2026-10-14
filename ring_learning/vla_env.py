"""V5 physical execution and a fixed tabletop RGB observation pass.

The policy receives seven own-proprioception values, never object coordinates,
teacher waypoints or simulator phases. Its fixed orthographic camera hides all
robot geometry in a declared special rendering pass; wide demonstration views
show the robot. Segmentation masks and true-state target scores are training /
evaluation utilities and must never be passed to the runtime controller.

The physical action prefix matches RingEnv.step. Stage-based transitions and
terminations are omitted: only an explicit controller finish or global timeout
ends an episode. Object-state scoring cannot schedule robot actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .sim import ACTION_SCALE, ACTION_SIZE, CONTROL_DT, RingEnv, _scene_xml


IMAGE_SIZE = 256
CAMERA_CENTER_XY = np.array([.47, 0.])
CAMERA_SPAN_M = .56
COMMANDS = (
    'put the ring inside the bearing',
    'put the ring left of the bearing',
    'put the ring right of the bearing',
)
TARGET_OFFSETS_X_M = (0., -.06, .06)
PROPRIO_SIZE = 7


def camera_signature():
    """Hash the static projection/rendering declaration, independent of scene."""
    specification = dict(name='tabletop', projection='orthographic', image_size=IMAGE_SIZE,
        center_xy_m=CAMERA_CENTER_XY.tolist(), height_m=.80, vertical_span_m=CAMERA_SPAN_M,
        horizontal_span_m=CAMERA_SPAN_M, xyaxes=[1, 0, 0, 0, 1, 0],
        pixel_coordinates='normalized raster edges; pixel centers (index+0.5)/256',
        world_to_pixel_axes=['positive X goes right', 'positive Y goes up'],
        robot_geometry_hidden=True, hidden_geom_group=5, crop_uses_object_coordinates=False,
        camera_motion='fixed in world', masks='geom segmentation; training only')
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':')).encode()
    return dict(specification=specification, sha256=hashlib.sha256(encoded).hexdigest())


def _vla_scene_xml():
    root = ET.fromstring(_scene_xml())
    for body in root.findall('.//body'):
        if body.get('name') not in ('ring', 'bearing'):
            for geom in body.findall('geom'):
                geom.set('group', '5')
    world = root.find('worldbody')
    ET.SubElement(world, 'camera', name='tabletop', mode='fixed', projection='orthographic',
                  pos='.47 0 .80', xyaxes='1 0 0 0 1 0', fovy=str(CAMERA_SPAN_M),
                  resolution=f'{IMAGE_SIZE} {IMAGE_SIZE}')
    return ET.tostring(root, encoding='unicode')


def _instruction_index(instruction):
    if isinstance(instruction, (int, np.integer)) and not isinstance(instruction, (bool, np.bool_)):
        if 0 <= instruction < len(COMMANDS):
            return int(instruction)
    if isinstance(instruction, str):
        aliases = {'inside': 0, 'left': 1, 'right': 2}
        if instruction in aliases:
            return aliases[instruction]
        if instruction in COMMANDS:
            return COMMANDS.index(instruction)
    raise ValueError('Unknown V5 instruction')


def _outside_target_valid(env, instruction_index):
    """True-state grading only; left/right have the same rest/release checks."""
    ring, bearing = env.ring_xyz, env.bearing_xyz
    target = bearing[:2]+[TARGET_OFFSETS_X_M[instruction_index], 0.]
    rot = env.data.xmat[env.ring_body_id].reshape(3, 3)
    tilt = math.acos(np.clip(abs(rot[2, 2]), 0, 1))
    velocity = env.data.qvel[env.ring_dof_adr:env.ring_dof_adr+6]
    return bool(np.linalg.norm(ring[:2]-target) < .003
        and tilt <= math.radians(8)
        and abs(ring[2]-(env.table_z+env.ring_height/2)) <= .0015
        and np.linalg.norm(velocity[:3]) < .010 and np.linalg.norm(velocity[3:]) < .15
        and env._released_command and env.finger_opening >= .0045
        and env.ring_finger_contacts() == 0)


def score_target(env, instruction):
    """Evaluate a declared text target from actual physics; not a policy input."""
    index = _instruction_index(instruction)
    if index == 0:
        return bool(env.success())
    return bool(env._target_settled_for[index] >= env.settle_duration
                and _outside_target_valid(env, index))


class VLAEnv(RingEnv):
    """Same contact physics; no oracle stages; seven own-proprioception values."""
    def __init__(self, *, render_width=640, render_height=480, max_steps=800):
        if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self._policy_renderer = None
        self._external_finished = False
        self._target_settled_for = np.zeros(3)
        # Base construction provides the unchanged IK buffers and servo schema.
        # The second compiled scene differs only in camera and render groups.
        super().__init__(render_width=render_width, render_height=render_height, max_steps=int(max_steps))
        self.model = mujoco.MjModel.from_xml_string(_vla_scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self._ring_geom_ids = np.flatnonzero(self.model.geom_bodyid == self.ring_body_id)
        self._bearing_geom_ids = np.flatnonzero(self.model.geom_bodyid == self.bearing_body_id)
        self._policy_option = mujoco.MjvOption()
        self._policy_option.geomgroup[:] = 1
        self._policy_option.geomgroup[5] = 0
        self._wide_option = mujoco.MjvOption()
        self._wide_option.geomgroup[:] = 1
        self.reset((.39, -.07), (.48, .02), seed=0)

    def reset(self, *args, **kwargs):
        self._external_finished = False
        self._target_settled_for = np.zeros(3)
        return super().reset(*args, **kwargs)

    def observe(self):
        tool = self.tool_xyz
        return np.r_[tool, self.command_xyz-tool, self.finger_opening].astype(np.float32)

    def finish(self):
        """Mark that the controller has finished; return the original task score."""
        self._external_finished = self._done = True
        if not self.success():
            self.failure_reason = 'controller_finished_without_success'
        return bool(self.success())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise ValueError('action must be finite length4')
        action = np.clip(action, -1, 1)
        self.command_xyz += ACTION_SCALE*action[:3]
        self.command_xyz = np.clip(self.command_xyz, [0.23, -0.30, self.table_z+0.0038], [0.70, 0.30, 0.40])
        self.joint_target = self._ik(self.command_xyz)
        self.data.ctrl[:7] = self.joint_target
        grip_target = self.closed_width+(action[3]+1)/2*(self.open_width-self.closed_width)
        self.data.ctrl[7] = grip_target
        self._released_command = bool(action[3] > .5)
        for _ in range(round(CONTROL_DT/self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        valid = (self._placement_valid(), _outside_target_valid(self, 1), _outside_target_valid(self, 2))
        self._target_settled_for = np.where(valid, self._target_settled_for+CONTROL_DT, 0.)
        self._settled_for = float(self._target_settled_for[0])
        self._done = bool(self._external_finished or self.steps >= self.max_steps)
        if self.steps >= self.max_steps and not self.success():
            self.failure_reason = 'timeout'
        info = dict(success=self.success(), failure_reason=self.failure_reason, steps=self.steps,
                    ring_xyz=self.ring_xyz.tolist(), tool_xyz=self.tool_xyz.tolist(),
                    settled_for=self._settled_for, finger_contacts=self.ring_finger_contacts(),
                    controller_finished=self._external_finished, stage_fsm_enabled=False)
        return self.observe(), self._done, info

    @staticmethod
    def project_xy(xy):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError('World XY must be finite with final dimension2')
        return (xy-CAMERA_CENTER_XY)*[1., -1.]/CAMERA_SPAN_M+.5

    @staticmethod
    def pixel_to_world(normalized_xy):
        nxy = np.asarray(normalized_xy, dtype=float)
        if nxy.ndim < 1 or nxy.shape[-1] != 2 or not np.isfinite(nxy).all():
            raise ValueError('Normalized pixel XY must be finite with final dimension2')
        return (nxy-.5)*[1., -1.]*CAMERA_SPAN_M+CAMERA_CENTER_XY

    def _update_policy_scene(self):
        if self._policy_renderer is None:
            self._policy_renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        self._policy_renderer.update_scene(self.data, camera='tabletop', scene_option=self._policy_option)

    def render_policy(self):
        """Fixed full-tabletop RGB; no segmentation, crops, or object-state reads."""
        self._update_policy_scene()
        return self._policy_renderer.render().copy()

    def render_masks(self):
        """Training labels from identical camera/visibility; never runtime inputs."""
        self._update_policy_scene()
        self._policy_renderer.enable_segmentation_rendering()
        try:
            segmentation = self._policy_renderer.render()
            geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            masks = [geom & np.isin(segmentation[..., 0], identifiers)
                     for identifiers in (self._ring_geom_ids, self._bearing_geom_ids)]
            return np.asarray(masks, dtype=bool)
        finally:
            self._policy_renderer.disable_segmentation_rendering()

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.renderer.update_scene(self.data, camera='overview', scene_option=self._wide_option)
        return self.renderer.render().copy()

    def close(self):
        if self._policy_renderer is not None:
            self._policy_renderer.close()
            self._policy_renderer = None
        super().close()
