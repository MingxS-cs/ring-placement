"""Train planar path shape directly from personally captured video positions.

Robot teachers do not supply these targets. Validation selects a checkpoint;
test trajectory errors are computed only after saving that frozen checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dataset import split_episodes, write_json
from .video_policy import ShapeNet, save_video_policy, source_signature
from .integrity import environment_signature


def arc_cell_weights(progress):
    """Trapezoidal arc cells integrate irregularly spaced tracked points.

    Extra samples at the same position do not gain extra source weight. This
    weights the original tracked/interpolated labels without manufacturing new
    observations by resampling the path.
    """
    progress = np.asarray(progress, np.float64)
    if (progress.ndim != 1 or len(progress) < 2 or not np.isfinite(progress).all()
            or progress[0] != 0 or progress[-1] != 1 or np.any(np.diff(progress) < 0)):
        raise ValueError("Arc progress must be finite, monotonic and endpoint anchored")
    return np.r_[(progress[1]-progress[0])/2,
                 (progress[2:]-progress[:-2])/2,
                 (progress[-1]-progress[-2])/2]


def prepare_samples(records, episodes, seed=20261014):
    """Split original source IDs before any canonical rotation or interpolation.

    Each original trajectory has unit total weight. Arc-cell widths remove the
    bias of unequal point density. Interpolated points receive 0.25 of observed
    label weight, before normalization within each original source.
    """
    known = {e["episode_id"] for e in episodes}
    unknown = set(records)-known
    if unknown:
        raise ValueError(f"Motion contains unknown source episode IDs: {sorted(unknown)}")
    splits = split_episodes(episodes, seed)
    membership = {name: sorted(e["episode_id"] for e in group) for name, group in splits.items()}
    prepared = {name: [] for name in splits}
    for name, group in splits.items():
        for episode in group:
            source = episode["episode_id"]
            if source not in records:
                continue
            record = records[source]
            if record.get("episode_id", source) != source:
                raise ValueError("Motion source key differs from its episode_id")
            if record.get("status", "accepted") != "accepted":
                continue
            path = record["path"]
            progress = np.asarray(path["progress"], np.float64)
            xy = np.asarray(path["relative_xy_m"], np.float64)
            if (progress.ndim != 1 or len(progress) < 3 or xy.shape != (len(progress), 2)
                    or not np.isfinite(progress).all() or not np.isfinite(xy).all()):
                raise ValueError("Motion points must be finite with matching progress and XY shapes")
            if (progress[0] != 0 or progress[-1] != 1 or np.any(np.diff(progress) < 0)
                    or np.any((progress < 0) | (progress > 1))):
                raise ValueError("Motion progress must be monotonic with exact zero/one endpoints")
            direction = xy[-1]-xy[0]
            length = float(np.linalg.norm(direction))
            if length < .005:
                raise ValueError("Video endpoint distance must be at least 5mm")
            longitudinal = direction/length
            basis = np.array([longitudinal, [-longitudinal[1], longitudinal[0]]])
            linear = (1-progress[:, None])*xy[0]+progress[:, None]*xy[-1]
            residual = (xy-linear) @ basis.T / length
            kinds = path.get("sample_kind", ["observed"]*len(progress))
            if len(kinds) != len(progress) or any(k not in ("observed", "interpolated") for k in kinds):
                raise ValueError("sample_kind must identify every observed/interpolated path point")
            if "sample_weights" in path:
                confidence = np.asarray(path["sample_weights"], float)
                if (confidence.shape != progress.shape or not np.isfinite(confidence).all()
                        or np.any(confidence <= 0)):
                    raise ValueError("Path evidence weights must be positive finite with one weight per point")
            # Contour confidence is an uncalibrated heuristic, retained only as
            # extraction evidence. A fixed kind multiplier defines the loss.
            weights = np.array([1. if k == "observed" else .25 for k in kinds])
            weights *= arc_cell_weights(progress)
            weights /= weights.sum()
            prepared[name].append({"source": source, "x": np.column_stack([progress, np.full(len(progress), length/.1)]).astype(np.float32),
                "y": residual.astype(np.float32), "weights": weights.astype(np.float32),
                "xy": xy, "start": xy[0], "goal": xy[-1], "length_m": length,
                "sample_kind": list(kinds)})
    if not prepared["train"] or not prepared["val"]:
        raise ValueError("Need accepted real trajectories in both training and validation splits")
    return prepared, membership


def _stack(group):
    return tuple(torch.from_numpy(np.concatenate([item[key] for item in group], axis=0))
                 for key in ("x", "y", "weights"))


def trajectory_metrics(model, group):
    """Source-uniform displacement error in approximate image-plane metres."""
    per_source = []
    for item in group:
        x = torch.from_numpy(item["x"])
        with torch.inference_mode():
            prediction = model(x).cpu().numpy()
        errors = np.linalg.norm(prediction-item["y"], axis=1)*item["length_m"]
        baseline = np.linalg.norm(item["y"], axis=1)*item["length_m"]
        weights = item["weights"]
        per_source.append({"source_episode_id": item["source"], "points": len(x),
            "observed_points": item["sample_kind"].count("observed"),
            "interpolated_points": item["sample_kind"].count("interpolated"),
            "network_mean_displacement_m": float(np.sum(errors*weights)),
            "straight_mean_displacement_m": float(np.sum(baseline*weights)),
            "network_max_displacement_m": float(np.max(errors)),
            "straight_max_displacement_m": float(np.max(baseline))})
    return {"sources": len(per_source),
        "network_mean_displacement_m": float(np.mean([r["network_mean_displacement_m"] for r in per_source])) if per_source else None,
        "straight_mean_displacement_m": float(np.mean([r["straight_mean_displacement_m"] for r in per_source])) if per_source else None,
        "per_source": per_source}


def train_motion(motion_path, real_path, output_dir, *, epochs=1200, hidden=32,
                 seed=20261014, learning_rate=.001, max_residual_m=.025):
    from .motion import load_motion
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    motion_path, real_path, output_dir = Path(motion_path), Path(real_path), Path(output_dir)
    real_payload = json.loads(real_path.read_text(encoding="utf-8"))
    motion_payload = json.loads(motion_path.read_text(encoding="utf-8"))
    source_sha = real_payload.get("source", {}).get("sha256")
    if (not isinstance(source_sha, str) or len(source_sha) != 64
            or any(c not in "0123456789abcdef" for c in source_sha)
            or motion_payload.get("source", {}).get("sha256") != source_sha
            or motion_payload.get("endpoint_manifest_sha256") != hashlib.sha256(real_path.read_bytes()).hexdigest()):
        raise ValueError("Motion provenance differs from the real video source or exact endpoint manifest")
    episodes = real_payload["episodes"]
    records = load_motion(motion_path)
    prepared, membership = prepare_samples(records, episodes, seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = ShapeNet(hidden, max_residual_m)
    xtrain, ytrain, wtrain = _stack(prepared["train"])
    xval, yval, wval = _stack(prepared["val"])
    # Source-uniform weighted statistics; never fitted on held-out trajectories.
    with torch.no_grad():
        mean = (xtrain*wtrain[:, None]).sum(0)/wtrain.sum()
        std = (((xtrain-mean).square()*wtrain[:, None]).sum(0)/wtrain.sum()).sqrt().clamp_min(.01)
        model.input_mean.copy_(mean)
        model.input_std.copy_(std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history, started = [], time.monotonic()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xtrain)
        train_loss = ((prediction-ytrain).square().sum(1)*wtrain).sum()/wtrain.sum()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(((model(xval)-yval).square().sum(1)*wval).sum()/wval.sum())
        history.append({"epoch": epoch+1, "train_loss": float(train_loss.detach()), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch+1
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch+1) % 100 == 0:
            print(f"epoch={epoch+1} train={float(train_loss.detach()):.6f} val={val_loss:.6f}", flush=True)
        if epoch+1-best_epoch >= 250:
            break
    model.load_state_dict(best_state)
    model.eval()
    metadata = {"schema_version": 1, "method": "real_video_supervised_planar_path_shape",
        "seed": seed, "device": "cpu", "architecture": f"2->{hidden}->{hidden}->2 ReLU, endpoint envelope, physical norm cap",
        "parameter_count": model.parameter_count, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "duration_s": time.monotonic()-started,
        "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "real_episodes_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "original_video_sha256": source_sha,
        "motion_endpoint_provenance_verified": True,
        "source_splits": membership,
        "used_sources": {name: [r["source"] for r in group] for name, group in prepared.items()},
        "training_sources": sorted(r["source"] for r in prepared["train"]),
        "validation_sources": sorted(r["source"] for r in prepared["val"]),
        "environment": environment_signature(), "implementation_sources": source_signature(),
        "training_points": len(xtrain), "validation_points": len(xval),
        "normalizer": {"mean": model.input_mean.tolist(), "std": model.input_std.tolist(), "fitted_on": "training sources only, source-uniform arc-cell quadrature times fixed observed/interpolated kind weights"},
        "inputs": ["arc_progress_0_to_1", "start_to_goal_distance_divided_by_0.1_m"],
        "outputs": ["endpoint_anchored_local_longitudinal_residual_divided_by_distance", "endpoint_anchored_local_lateral_residual_divided_by_distance"],
        "max_residual_m": max_residual_m, "interpolated_relative_weight": .25,
        "sample_weighting": "Irregular observed/gap-interpolated points weighted by trapezoidal arc-cell width times label weight, then normalized to unit total weight per source; no uniform path resampling",
        "contour_confidence": "Uncalibrated extraction heuristic retained as evidence; not used as a training or evaluation loss weight",
        "label_source": "tracked_and_gap_interpolated_real_phone_video_ring_XY_not_robot_teacher",
        "endpoint_policy": "learn shape relative to observed start/end line; robot retargets exact end to bearing center",
        "retiming": "Arc-length progress and 40mm/s robot command speed; original human timing is not learned",
        "rotation_translation_augmentation": "Analytical local-frame equivariance; no extra independent source recordings",
        "runtime_inputs": "start/goal/progress only; held-out motion paths never read by controller",
        "hybrid_controller": "Neural XY in transfer; geometric XY/Z/gripper in other phases; geometric Z/gripper in transfer",
        "no_test_data_used_for_model_selection": True,
        "limitations": ["Approximate uncalibrated image-plane XY; not 3D pose", "Interpolated occlusions are uncertain labels", "Human grasp/release/height/orientation are not learned", "Test sources are from same concatenated recording", "No real robot validation"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir/"policy.pt"
    # Freeze the selected model before computing any held-out path metrics.
    save_video_policy(checkpoint_path, model, metadata)
    metrics = {name: trajectory_metrics(model, group) for name, group in prepared.items()}
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    write_json(output_dir/"training.json", {"metadata": metadata, "history": history,
        "frozen_checkpoint_sha256": checkpoint_sha256, "trajectory_metrics": metrics,
        "metric_interpretation": "Approximate XY displacement against extracted/interpolated human paths; not physical localization accuracy"})
    print(json.dumps({"parameter_count": model.parameter_count, "best_epoch": best_epoch,
        "checkpoint_sha256": checkpoint_sha256, "trajectory_metrics": {k: {n: v for n, v in value.items() if n != "per_source"} for k, value in metrics.items()}}, indent=2), flush=True)
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default="data/real_motion.json")
    parser.add_argument("--real", default="data/real_episodes.json")
    parser.add_argument("--output", default="outputs/video_model")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    train_motion(args.motion, args.real, args.output, epochs=args.epochs, hidden=args.hidden, seed=args.seed)


if __name__ == "__main__":
    main()
