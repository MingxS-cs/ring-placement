"""Proprioceptive execution of neural Cartesian/gripper action chunks."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .sim import ACTION_SCALE
from .vla_policy import encode_text


HOLD_STEPS = np.array([4,4,20,4]+[1]*16+[4,12,10])


class VLAController:
    """Single initial image/text inference; generic waypoint tracking thereafter.

    No object pose, teacher, scoring result, or environment phase is accessed.
    Scheduling and hold durations are hand-coded execution conventions.
    """
    def __init__(self, model, instruction, *, ablation='none'):
        if ablation not in ('none','black_image'):
            raise ValueError('Unknown image ablation')
        self.model, self.instruction, self.ablation = model, instruction, ablation
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')
        self.tokens = encode_text([instruction],device=self.device)
        self.predicted_waypoints = None
        self.waypoint_index = self.neural_forward_calls = self._hold = 0
        self.finished = False
        self.input_image_sha256 = None

    def __call__(self, env, observation=None):
        if self.predicted_waypoints is None:
            rgb = env.render_policy()
            if self.ablation == 'black_image':
                rgb = np.zeros_like(rgb)
            self.input_image_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
            images = torch.tensor(rgb.transpose(2,0,1).copy(),device=self.device,dtype=torch.float32)[None]/255.
            self.model.eval()
            with torch.inference_mode():
                output, _ = self.model(images,self.tokens)
            self.neural_forward_calls += 1
            self.predicted_waypoints = output[0].detach().cpu().numpy()
            if self.predicted_waypoints.shape != (23,4) or not np.isfinite(self.predicted_waypoints).all():
                raise ValueError('Neural action chunk must contain 23 finite XYZ/gripper targets')
            xyz = self.predicted_waypoints[:,:3]
            if np.any(xyz < [.23,-.30,.1038]) or np.any(xyz > [.70,.30,.40]):
                raise ValueError('Neural action chunk is outside the physical workspace')
        index = min(self.waypoint_index,22)
        target = self.predicted_waypoints[index]
        delta = target[:3]-env.command_xyz
        norm = float(np.linalg.norm(delta))
        if norm > ACTION_SCALE:
            delta *= ACTION_SCALE/norm
        if np.linalg.norm(env.tool_xyz-target[:3]) < .0014:
            self._hold += 1
            if self._hold >= HOLD_STEPS[index]:
                self.waypoint_index += 1
                self._hold = 0
                if self.waypoint_index >= 23:
                    self.finished = True
        else:
            self._hold = 0
        return np.r_[delta/ACTION_SCALE,np.clip(target[3],-1.,1.)].astype(np.float32)
