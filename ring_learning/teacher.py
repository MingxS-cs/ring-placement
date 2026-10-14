"""Geometric demonstration labels using the same physical low-level controller."""
from __future__ import annotations
import numpy as np
from .sim import ACTION_SCALE, RingEnv


def teacher(env: RingEnv):
    delta = np.clip((env.phase_target() - env.command_xyz) / ACTION_SCALE, -1, 1)
    if env.phase == 3:
        delta[:2] = 0.0  # Lift vertically from the current commanded grasp pose.
    grip = -1.0 if env.phase in (2, 3, 4, 5) else 1.0
    return np.r_[delta, grip].astype(np.float32)


def rollout(env: RingEnv, max_steps=450, *, render_every=0):
    observations, actions, phases, frames = [], [], [], []
    max_height, contact_steps = float(env.ring_xyz[2]), 0
    info = {}
    for step in range(max_steps):
        observations.append(env.observe())
        phases.append(env.phase)
        action = teacher(env)
        actions.append(action)
        if render_every and step % render_every == 0:
            frames.append(env.render())
        _, done, info = env.step(action)
        max_height = max(max_height, float(env.ring_xyz[2]))
        contact_steps += int(env.ring_finger_contacts() > 0)
        if done:
            break
    return {"obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "phases": np.asarray(phases, dtype=np.int64), "success": env.success(),
            "info": info, "frames": frames, "max_ring_height": max_height,
            "finger_contact_steps": contact_steps}
