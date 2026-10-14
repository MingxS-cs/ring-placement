"""Video-supervised planar path shape, with explicit geometric grasp/release.

The network does not recover depth, orientation, force, human timing, or joint
angles. Its local residual is an endpoint-anchored XY shape. The simulator's
existing Cartesian servo, IK, contact physics, and phase guards execute it.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn

from .sim import ACTION_SCALE, CONTROL_DT
from .teacher import teacher
from .integrity import environment_signature


class ShapeNet(nn.Module):
    """Two inputs (arc progress, endpoint distance/0.1m) and two local residuals.

    Output residuals are divided by endpoint distance. The 4p(1-p) envelope
    anchors both endpoints exactly. The physical residual norm is capped at
    25mm, independently of the motion scale, before robot execution.
    """

    def __init__(self, hidden=32, max_residual_m=.025):
        super().__init__()
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        if not np.isfinite(max_residual_m) or max_residual_m <= 0:
            raise ValueError("max_residual_m must be finite and positive")
        self.hidden, self.max_residual_m = hidden, float(max_residual_m)
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.register_buffer("input_mean", torch.zeros(2))
        self.register_buffer("input_std", torch.ones(2))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, features):
        predicted = self.net((features-self.input_mean)/self.input_std)
        progress, length = features[..., :1], features[..., 1:2]*.1
        residual = predicted * (4*progress*(1-progress))
        norm_m = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)*length
        return residual * torch.clamp(self.max_residual_m/norm_m.clamp_min(1e-12), max=1.)

    def predict_residual(self, progress, length_m):
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and between zero and one")
        if not np.isfinite(length_m) or length_m < 0:
            raise ValueError("length must be finite and nonnegative")
        if length_m == 0 or progress in (0, 1):
            return np.zeros(2)
        features = torch.tensor([[progress, length_m/.1]], dtype=torch.float32,
                                device=self.input_mean.device)
        with torch.inference_mode():
            result = self(features).detach().cpu().numpy()[0].astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Network produced a nonfinite residual")
        return result

    def waypoint(self, start_xy, goal_xy, progress):
        start, goal = np.asarray(start_xy, float), np.asarray(goal_xy, float)
        if start.shape != (2,) or goal.shape != (2,) or not np.isfinite(np.r_[start, goal]).all():
            raise ValueError("start and goal must be finite XY positions")
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and between zero and one")
        if progress == 0:
            return start.copy()
        if progress == 1:
            return goal.copy()
        direction = goal-start
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return start.copy()
        longitudinal = direction/length
        basis = np.array([longitudinal, [-longitudinal[1], longitudinal[0]]])
        residual_world = self.predict_residual(progress, length) @ basis * length
        # A final float64 cap avoids floating-point overshoot of the SI bound.
        norm = np.linalg.norm(residual_world)
        if norm > self.max_residual_m:
            residual_world *= self.max_residual_m/norm
        return (1-progress)*start + progress*goal + residual_world


def save_video_policy(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "hidden": model.hidden,
                "max_residual_m": model.max_residual_m,
                "state_dict": model.state_dict(), "metadata": metadata}, path)


def source_signature():
    """Normalized Python bytes survive Git line-ending conversion on Windows."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/f"{name}.py").read_text(encoding="utf-8-sig")
        .replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()
        for name in ("video_policy", "video_train")}


def load_video_policy(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported video policy schema")
    metadata = value["metadata"]
    if "environment" in metadata and metadata["environment"] != environment_signature():
        raise ValueError("Video checkpoint environment differs from current simulator/teacher")
    if "implementation_sources" in metadata and metadata["implementation_sources"] != source_signature():
        raise ValueError("Video checkpoint implementation source differs from current policy/training source")
    model = ShapeNet(value["hidden"], value["max_residual_m"])
    model.load_state_dict(value["state_dict"])
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Video policy contains nonfinite weights")
    if not (model.input_std > 0).all():
        raise ValueError("Video policy normalizer standard deviations must be positive")
    model.eval()
    return model, metadata


class VideoController:
    """Neural XY transfer only; other seven phases use the geometric controller.

    The callable never opens the real-motion dataset. Each episode supplies
    only the simulator's commanded transfer-entry XY and task goal. Geometry
    ablation uses the identical progress schedule, speed cap, Z and gripper.
    The 40mm/s retiming is a robot convention, not measured human timing.
    """

    def __init__(self, model, *, geometry=False, max_step_m=.002):
        if not np.isfinite(max_step_m) or not 0 < max_step_m <= ACTION_SCALE:
            raise ValueError("max_step_m must be positive and no greater than ACTION_SCALE")
        self.model, self.geometry, self.max_step_m = model, bool(geometry), float(max_step_m)
        self.learned_steps = self.planned_steps = self.transfer_steps = 0
        self.progress, self.start_xy, self.goal_xy = 0., None, None
        self._previous_steps, self._previous_phase = None, None
        self.transfer_speed_m_s = self.max_step_m/CONTROL_DT

    def __call__(self, env, observation=None):
        if self._previous_steps is not None and env.steps <= self._previous_steps:
            self.progress, self.start_xy, self.goal_xy = 0., None, None
            self._previous_phase = None
        if env.phase != 4:
            self.planned_steps += 1
            action = teacher(env)
        else:
            target = np.asarray(env.phase_target(), float)
            if self.start_xy is None or self._previous_phase != 4:
                self.start_xy = np.asarray(env.command_xyz[:2], float).copy()
                self.goal_xy = target[:2].copy()
                self.progress = 0.
            length = float(np.linalg.norm(self.goal_xy-self.start_xy))
            self.progress = min(1., self.progress + self.max_step_m/max(length, self.max_step_m))
            if self.geometry:
                xy = (1-self.progress)*self.start_xy + self.progress*self.goal_xy
                self.planned_steps += 1
            else:
                xy = self.model.waypoint(self.start_xy, self.goal_xy, self.progress)
                self.learned_steps += 1
            delta = np.r_[xy, target[2]] - env.command_xyz
            norm = float(np.linalg.norm(delta))
            if norm > self.max_step_m:
                delta *= self.max_step_m/norm
            action = np.r_[delta/ACTION_SCALE, -1.].astype(np.float32)
            self.transfer_steps += 1
        self._previous_steps, self._previous_phase = env.steps, env.phase
        return action
