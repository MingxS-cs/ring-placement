"""Contact-based Panda task. Object state is written only by reset.

The video contributes planar configurations; tool height, downward orientation,
fingertips, IK and eight phase transitions are planned simulation components.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PHASES = ("approach", "descend", "grasp", "lift", "transfer", "lower", "release", "retract")
OBS_SIZE = 30
ACTION_SIZE = 4
ACTION_SCALE = 0.003
CONTROL_DT = 0.05
ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "vendor" / "panda"


def _annulus(asset, body, name, inner, outer, height, segments, rgba, mass=0.025):
    """Union of convex eight-vertex wedges; each hole stays physically empty."""
    for i in range(segments):
        angles = (2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments)
        vertices = [(r * math.cos(a), r * math.sin(a), z)
                    for z in (-height / 2, height / 2)
                    for r in (inner, outer) for a in angles]
        mesh_name = f"{name}_wedge_{i}"
        ET.SubElement(asset, "mesh", name=mesh_name,
                      vertex=" ".join(str(v) for vertex in vertices for v in vertex))
        ET.SubElement(body, "geom", name=f"{name}_{i}", type="mesh", mesh=mesh_name,
                      mass=str(mass / segments), rgba=rgba, friction="1.2 0.005 0.0001",
                      condim="4", solref="0.003 1", solimp="0.95 0.99 0.001", group="1")


def _scene_xml():
    root = ET.parse(PANDA_DIR / "panda.xml").getroot()
    root.set("model", "panda_ring_placement")
    root.find("compiler").set("meshdir", str(PANDA_DIR / "assets"))
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("iterations", "80")
    option.set("cone", "elliptic")
    ET.SubElement(root, "size", njmax="4000", nconmax="1500")
    for key in root.findall("keyframe"):
        root.remove(key)
    for body in root.findall(".//body"):
        body.set("gravcomp", "1")
    for default in root.findall(".//default[@class='collision']/geom"):
        default.set("contype", "2")
        default.set("conaffinity", "1")
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tool", pos="0 0 0.1084", size="0.002", rgba="1 0 0 0")
    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        for geom in list(body.findall("geom")):
            body.remove(geom)
        # Replaces bulky original fingers with slender custom extensions.
        ET.SubElement(body, "geom", name=f"{name}_stem", type="box",
                      pos="0 0 0.023", size="0.0015 0.001 0.023", mass="0.008",
                      rgba="0.25 0.27 0.3 1", contype="4", conaffinity="1")
        ET.SubElement(body, "geom", name=f"{name}_tip", type="box",
                      pos="0 0 0.050", size="0.0015 0.001 0.004", mass="0.002",
                      rgba="0.07 0.08 0.1 1", friction="2.5 0.01 0.0001", condim="4",
                      contype="4", conaffinity="1", solref="0.003 1", solimp="0.95 0.99 0.001")
    actuator = root.find("actuator")
    old_grip = actuator.find("general[@name='actuator8']")
    actuator.remove(old_grip)
    # Position servo on coupled tendon, half opening in metres.
    ET.SubElement(actuator, "general", name="gripper", tendon="split", dyntype="none",
                  biastype="affine", gainprm="1800", biasprm="0 -1800 -5",
                  ctrlrange="0 0.008", forcerange="-12 12")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="2 2 0.1",
                  rgba="0.85 0.87 0.90 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="table", type="box", pos="0.48 0 0.05",
                  size="0.36 0.40 0.05", rgba="0.52 0.38 0.25 1", friction="0.7 0.005 0.0001")
    ET.SubElement(world, "camera", name="overview", pos="0.95 -0.75 0.8",
                  xyaxes="0.78 0.63 0 -0.40 0.49 0.77", fovy="42")
    ET.SubElement(world, "light", pos="0.5 -0.3 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    asset = root.find("asset")
    bearing = ET.SubElement(world, "body", name="bearing", pos="0.48 0.02 0.1025")
    _annulus(asset, bearing, "bearing", 0.020, 0.025, 0.005, 64, "0.70 0.73 0.77 1")
    ring = ET.SubElement(world, "body", name="ring", pos="0.39 -0.07 0.1035")
    ET.SubElement(ring, "freejoint", name="ring_free")
    _annulus(asset, ring, "ring", 0.010, 0.0125, 0.007, 48, "0.89 0.59 0.10 1")
    return ET.tostring(root, encoding="unicode")


class RingEnv:
    """Normalized xyz delta + gripper action; fixed downward Cartesian tool.

    Observation order: tool/ring/bearing xyz, ring-tool xyz, bearing-ring xyz,
    tool velocity xyz, ring velocity xyz, half-opening, eight phase indicators.
    SI units, float32, length30. No teacher waypoint is present in observations.
    """
    table_z = 0.10
    ring_outer_radius = 0.0125
    bearing_inner_radius = 0.020
    ring_height = 0.007
    grip_offset = np.array([0.0, 0.01125, 0.0008])
    open_width = 0.006
    closed_width = 0.00025
    settle_duration = 0.40

    def __init__(self, *, render_width=640, render_height=480, max_steps=450):
        self.model = mujoco.MjModel.from_xml_string(_scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.ring_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ring")
        self.ring_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ring_free")
        self.ring_qpos_adr = self.model.jnt_qposadr[self.ring_joint_id]
        self.ring_dof_adr = self.model.jnt_dofadr[self.ring_joint_id]
        self.bearing_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bearing")
        self.tool_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool")
        self.finger_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                           for name in ("left_finger", "right_finger")}
        self.render_width, self.render_height = render_width, render_height
        self.renderer = None
        self.max_steps = max_steps
        self.target_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.reset((0.39, -0.07), (0.48, 0.02), seed=0)

    @property
    def tool_xyz(self):
        return self.data.site_xpos[self.tool_site_id].copy()

    @property
    def ring_xyz(self):
        return self.data.xpos[self.ring_body_id].copy()

    @property
    def bearing_xyz(self):
        return self.data.xpos[self.bearing_body_id].copy()

    @property
    def phase_name(self):
        return PHASES[self.phase]

    @property
    def finger_opening(self):
        return float(np.mean(self.data.qpos[7:9]))

    def reset(self, ring_xy=(0.39, -0.07), bearing_xy=(0.48, 0.02), seed=0):
        ring_xy, bearing_xy = np.asarray(ring_xy, dtype=float), np.asarray(bearing_xy, dtype=float)
        if ring_xy.shape != (2,) or bearing_xy.shape != (2,) or not np.all(np.isfinite(np.r_[ring_xy, bearing_xy])):
            raise ValueError("ring_xy and bearing_xy must be finite length2 positions in metres")
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.bearing_body_id] = np.r_[bearing_xy, self.table_z + 0.0025]
        self.data.qpos[:9] = [0, -0.5, 0, -2.1, 0, 1.6, 0.7854, self.open_width, self.open_width]
        self.data.qpos[self.ring_qpos_adr:self.ring_qpos_adr+7] = np.r_[ring_xy, self.table_z + 0.0035, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.initial_ring_xyz = self.ring_xyz
        self.grasp_xyz = self.initial_ring_xyz + self.grip_offset
        self.goal_xyz = np.r_[bearing_xy, self.table_z + 0.0035] + self.grip_offset
        self.phase = 0
        self.phase_steps = 0
        self.steps = 0
        self._settled_for = 0.0
        self._done = False
        self.failure_reason = ""
        self._released_command = True
        # Fixed workspace home: no reset oracle prealigns the tool above the ring.
        self.command_xyz = np.array([0.45, -0.02, 0.25])
        if hasattr(self, "joint_target"):
            del self.joint_target
        self.data.qpos[:7] = self._ik(self.command_xyz, iterations=100)
        self.data.ctrl[:7] = self.data.qpos[:7]
        self.data.ctrl[7] = self.open_width
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def _ik(self, target, iterations=12):
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        if hasattr(self, "joint_target"):
            d.qpos[:7] = self.joint_target
        for _ in range(iterations):
            mujoco.mj_forward(self.model, d)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, d.site_xmat[self.tool_site_id])
            rot_error = np.empty(3)
            mujoco.mju_subQuat(rot_error, self.target_quat, quat)
            rot_error = d.site_xmat[self.tool_site_id].reshape(3, 3) @ rot_error
            error = np.r_[np.asarray(target) - d.site_xpos[self.tool_site_id], rot_error]
            if np.linalg.norm(error) < 1e-6:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.tool_site_id)
            jac = np.vstack([self._jacp[:, :7], self._jacr[:, :7]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            d.qpos[:7] += np.clip(dq, -0.12, 0.12)
            d.qpos[:7] = np.clip(d.qpos[:7], self.model.jnt_range[:7, 0]+0.001,
                               self.model.jnt_range[:7, 1]-0.001)
        return d.qpos[:7].copy()

    def phase_target(self):
        if self.phase == 0:
            return self.grasp_xyz + [0, 0, 0.065]
        if self.phase in (1, 2):
            return self.grasp_xyz.copy()
        if self.phase == 3:
            return self.grasp_xyz + [0, 0, 0.060]
        if self.phase == 4:
            return self.goal_xyz + [0, 0, 0.060]
        if self.phase in (5, 6):
            return self.goal_xyz.copy()
        return self.goal_xyz + [0, 0, 0.090]

    def ring_finger_contacts(self):
        count = 0
        for c in self.data.contact:
            a, b = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (a == self.ring_body_id and b in self.finger_ids) or (b == self.ring_body_id and a in self.finger_ids):
                count += 1
        return count

    def observe(self):
        tool, ring, bearing = self.tool_xyz, self.ring_xyz, self.bearing_xyz
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.tool_site_id)
        tool_velocity = self._jacp @ self.data.qvel
        ring_velocity = self.data.qvel[self.ring_dof_adr:self.ring_dof_adr+3]
        phase = np.zeros(len(PHASES)); phase[self.phase] = 1
        return np.r_[tool, ring, bearing, ring-tool, bearing-ring,
                     tool_velocity, ring_velocity, self.finger_opening, phase].astype(np.float32)

    def _placement_valid(self):
        ring, bearing = self.ring_xyz, self.bearing_xyz
        rot = self.data.xmat[self.ring_body_id].reshape(3, 3)
        tilt = math.acos(np.clip(abs(rot[2, 2]), 0, 1))
        # Sample both circumference end planes for actual horizontal projection.
        theta = np.linspace(0, 2*math.pi, 128, endpoint=False)
        vertices = np.column_stack([self.ring_outer_radius*np.cos(theta),
                                   self.ring_outer_radius*np.sin(theta), np.zeros(len(theta))])
        projection = []
        for z in (-self.ring_height/2, self.ring_height/2):
            vertices[:, 2] = z
            projection.append((vertices @ rot.T + ring - bearing)[:, :2])
        radius = np.max(np.linalg.norm(np.vstack(projection), axis=1))
        velocity = self.data.qvel[self.ring_dof_adr:self.ring_dof_adr+6]
        return bool(radius <= self.bearing_inner_radius - 0.0003
                    and tilt <= math.radians(8)
                    and abs(ring[2] - (self.table_z+self.ring_height/2)) <= 0.0015
                    and np.linalg.norm(velocity[:3]) < 0.010
                    and np.linalg.norm(velocity[3:]) < 0.15
                    and self._released_command and self.finger_opening >= 0.0045
                    and self.ring_finger_contacts() == 0)

    def success(self):
        return bool(self._settled_for >= self.settle_duration and self._placement_valid())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be finite length4")
        action = np.clip(action, -1, 1)
        self.command_xyz += ACTION_SCALE * action[:3]
        self.command_xyz = np.clip(self.command_xyz, [0.23, -0.30, self.table_z+0.0038], [0.70, 0.30, 0.40])
        self.joint_target = self._ik(self.command_xyz)
        self.data.ctrl[:7] = self.joint_target
        grip_target = self.closed_width + (action[3]+1)/2 * (self.open_width-self.closed_width)
        self.data.ctrl[7] = grip_target
        self._released_command = bool(action[3] > 0.5)
        for _ in range(round(CONTROL_DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.phase_steps += 1
        self._settled_for = self._settled_for + CONTROL_DT if self._placement_valid() else 0.0
        error = np.linalg.norm(self.tool_xyz - self.phase_target())
        stage_error = abs(self.tool_xyz[2] - self.phase_target()[2]) if self.phase == 3 else error
        if self.phase in (0, 1, 3, 4, 5) and stage_error < 0.0018 and self.phase_steps >= 4:
            if self.phase == 3 and (self.ring_xyz[2] < self.table_z+0.025 or self.ring_finger_contacts() == 0):
                self.failure_reason, self._done = "failed_lift", True
            else:
                self.phase += 1; self.phase_steps = 0
        elif self.phase == 2 and self.phase_steps >= 20:
            if self.ring_finger_contacts() > 0:
                self.phase += 1; self.phase_steps = 0
            elif self.phase_steps > 60:
                self.failure_reason, self._done = "failed_grasp", True
        elif self.phase == 6 and self.phase_steps >= 12 and self._released_command:
            self.phase += 1; self.phase_steps = 0
        elif self.phase == 7 and error < 0.002 and self.phase_steps >= 10:
            self._done = True
            if not self.success():
                self.failure_reason = "placement_not_settled_or_contained"
        if self.steps >= self.max_steps:
            self._done = True
            if not self.success():
                self.failure_reason = "timeout_" + self.phase_name
        info = {"success": self.success(), "phase": self.phase_name, "phase_index": self.phase,
                "failure_reason": self.failure_reason, "steps": self.steps,
                "ring_xyz": self.ring_xyz.tolist(), "tool_xyz": self.tool_xyz.tolist(),
                "settled_for": self._settled_for, "finger_contacts": self.ring_finger_contacts()}
        return self.observe(), self._done, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
