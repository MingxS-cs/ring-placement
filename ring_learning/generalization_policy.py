"""Small endpoint-anchored ridge model of optical human transfer paths.

Direction is measured in an explicit camera/task reference frame. Translating
the scene preserves predictions. Rotating the scene preserves predictions only
when ``reference_angle`` rotates with it; fixed-frame direction changes can
change the learned style. No depth, timing, force or grasp labels are recovered.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .integrity import environment_signature


MAX_RESIDUAL_M = .025


def progress_basis(progress, degree):
    """Endpoint envelope with one or two bounded progress terms."""
    p = np.asarray(progress, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Progress must be a finite vector in [0, 1]")
    if degree not in (0, 1) or isinstance(degree, bool):
        raise ValueError("Progress degree must be zero or one")
    basis = np.ones((len(p), degree+1))
    if degree == 1:
        basis[:, 1] = 2*p-1
    return (4*p*(1-p))[:, None] * basis


def reference_direction(direction_xy, reference_angle=0.):
    direction = np.asarray(direction_xy, dtype=np.float64)
    if (direction.shape != (2,) or not np.isfinite(direction).all()
            or not np.isfinite(reference_angle)):
        raise ValueError("Direction and reference angle must be finite")
    length = float(np.hypot(*direction))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Direction must have finite positive length")
    u = direction/length
    c, s = np.cos(reference_angle), np.sin(reference_angle)
    return np.array([c*u[0]+s*u[1], -s*u[0]+c*u[1]])


class DirectionRidgePolicy:
    """Six or twelve learned coefficients, predicting local SI displacement."""

    def __init__(self, coefficients, direction_mean, direction_std, *, degree=0,
                 alpha=.2, max_residual_m=MAX_RESIDUAL_M):
        progress_basis(np.array([0., 1.]), degree)
        self.degree, self.alpha = int(degree), float(alpha)
        self.coefficients = np.array(coefficients, dtype=np.float64, copy=True)
        self.direction_mean = np.array(direction_mean, dtype=np.float64, copy=True)
        self.direction_std = np.array(direction_std, dtype=np.float64, copy=True)
        self.max_residual_m = float(max_residual_m)
        if (self.coefficients.shape != (3*(degree+1), 2)
                or self.direction_mean.shape != (2,) or self.direction_std.shape != (2,)
                or not np.isfinite(self.coefficients).all()
                or not np.isfinite(self.direction_mean).all()
                or not np.isfinite(self.direction_std).all()
                or np.any(self.direction_std <= 0)
                or not np.isfinite(self.alpha) or self.alpha <= 0
                or not np.isfinite(self.max_residual_m) or self.max_residual_m <= 0
                or self.max_residual_m > MAX_RESIDUAL_M):
            raise ValueError("Ridge parameters must have valid finite shapes and bounds")

    @property
    def parameter_count(self):
        return int(self.coefficients.size)

    def design_matrix(self, progress, direction_in_reference):
        direction = np.asarray(direction_in_reference, dtype=np.float64)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("Reference-frame direction must be finite XY")
        geometry = np.r_[1., (direction-self.direction_mean)/self.direction_std]
        if not np.isfinite(geometry).all():
            raise ValueError("Direction normalizer produced nonfinite features")
        return (progress_basis(progress, self.degree)[:, :, None]
                * geometry[None, None, :]).reshape(-1, 3*(self.degree+1))

    def local_residuals(self, progress, direction_in_reference):
        """Return capped longitudinal/lateral residuals in metres."""
        design = self.design_matrix(progress, direction_in_reference)
        scale = float(np.max(np.abs(self.coefficients)))
        if scale == 0:
            return np.zeros((len(design), 2))
        # Scale before the dot product so large finite coefficients can still
        # saturate safely without producing an infinite vector norm.
        scaled = design @ (self.coefficients/scale)
        result = np.zeros_like(scaled)
        for index, row in enumerate(scaled):
            norm = float(np.hypot(*row))
            if norm:
                if scale > self.max_residual_m/norm:
                    result[index] = row/norm*self.max_residual_m
                else:
                    result[index] = row*scale
        if not np.isfinite(result).all():
            raise ValueError("Ridge model produced a nonfinite residual")
        return result

    def waypoint(self, start_xy, goal_xy, progress, reference_angle=0.):
        start, goal = np.asarray(start_xy, float), np.asarray(goal_xy, float)
        if (start.shape != (2,) or goal.shape != (2,)
                or not np.isfinite(np.r_[start, goal]).all()
                or not np.isfinite(progress) or not 0 <= progress <= 1
                or not np.isfinite(reference_angle)):
            raise ValueError("Waypoint geometry, progress and reference angle must be finite and valid")
        if progress == 0:
            return start.copy()
        if progress == 1:
            return goal.copy()
        direction = goal-start
        length = float(np.hypot(*direction))
        if not np.isfinite(length):
            raise ValueError("Endpoint displacement must have finite length")
        if length < 1e-12:
            return start.copy()
        longitudinal = direction/length
        basis = np.array([longitudinal, [-longitudinal[1], longitudinal[0]]])
        u = reference_direction(direction, reference_angle)
        residual = self.local_residuals(np.array([progress]), u)[0] @ basis
        # Float64 rotation can add one rounding unit to the fixed SI cap.
        norm = float(np.hypot(*residual))
        if norm > self.max_residual_m:
            residual *= self.max_residual_m/norm
        return (1-progress)*start + progress*goal + residual


def source_signature():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/f"{name}.py").read_text(encoding="utf-8-sig")
            .replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()
            for name in ("generalization_policy", "generalization_train")}


def save_generalization_policy(path, model, metadata):
    payload = dict(schema_version=1, model=dict(coefficients=model.coefficients.tolist(),
        direction_mean=model.direction_mean.tolist(), direction_std=model.direction_std.tolist(),
        degree=model.degree, alpha=model.alpha, max_residual_m=model.max_residual_m), metadata=metadata)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def load_generalization_policy(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported generalization policy schema")
    metadata = value["metadata"]
    if "environment" in metadata and metadata["environment"] != environment_signature():
        raise ValueError("Generalization checkpoint environment differs from current simulator/teacher")
    if "implementation_sources" in metadata and metadata["implementation_sources"] != source_signature():
        raise ValueError("Generalization checkpoint implementation source differs from current source")
    model = DirectionRidgePolicy(**value["model"])
    return model, metadata
