"""Train a locked small ridge family on development-source optical paths only.

Original source membership is frozen before path parsing or interpolation.
Validation is diagnostic and never selects a model; test paths are not parsed.
The targets are approximate image-plane metres, not recovered 3D robot actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import split_episodes
from .generalization_policy import (DirectionRidgePolicy, MAX_RESIDUAL_M,
    reference_direction, save_generalization_policy, source_signature)
from .integrity import environment_signature


SEED = 20261014
CANDIDATE_GRID = tuple((degree, alpha) for degree in (0, 1) for alpha in (.02, .2, 2.))
SIMPLICITY_TOLERANCE_M = .0001
MIN_DIRECTION_STD = .1


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _records(payload):
    records = payload.get("episodes", payload.get("records", []))
    if isinstance(records, dict):
        keyed_records = []
        for key, value in records.items():
            if not isinstance(value, dict) or value.get("episode_id", key) != key:
                raise ValueError("Keyed trajectory source differs from its episode ID")
            keyed_records.append(dict(value, episode_id=key))
        records = keyed_records
    if not isinstance(records, list):
        raise ValueError("Trajectory records must be a list or keyed mapping")
    result = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("episode_id"), str):
            raise ValueError("Every record requires a source episode ID")
        source = record["episode_id"]
        if source in result:
            raise ValueError("Duplicate trajectory source episode ID")
        result[source] = record
    return result


def _frame_array(values, message):
    try:
        frames = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(message) from error
    if (frames.ndim != 1 or not np.isfinite(frames).all()
            or np.any(frames != np.floor(frames)) or np.any(np.diff(frames) <= 0)):
        raise ValueError(message)
    return frames


def _prepare_record(record, episode, origin):
    source = episode["episode_id"]
    if record.get("status", "accepted") != "accepted":
        return None
    path = record.get("path")
    if not isinstance(path, dict):
        raise ValueError("An accepted trajectory requires a derived path")
    try:
        progress = np.asarray(path["progress"], dtype=float)
        xy = np.asarray(path["relative_xy_m"], dtype=float)
        frames = _frame_array(path["sample_source_frames"], "Path source frames must be finite increasing integers")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid finite trajectory path for {source}: {error}") from error
    if (progress.ndim != 1 or len(progress) < 3 or xy.shape != (len(progress), 2)
            or frames.shape != progress.shape or not np.isfinite(progress).all()
            or not np.isfinite(xy).all() or progress[0] != 0 or progress[-1] != 1
            or np.any(np.diff(progress) < 0) or np.any((progress < 0) | (progress > 1))):
        raise ValueError("Trajectory XY and progress must be finite, monotonic and endpoint anchored")
    first, last = episode.get("source_frame_start"), episode.get("source_frame_terminal")
    if (not isinstance(first, int) or not isinstance(last, int) or first >= last
            or np.any(frames < first) or np.any(frames > last)):
        raise ValueError("Trajectory source frames exceed original episode temporal bounds")
    for key, expected in (("source_frame_start", first), ("source_frame_terminal", last)):
        if key in record and record[key] != expected:
            raise ValueError("Trajectory record temporal bounds differ from original source episode")
    samples = record.get("samples", record.get("samples_manual_visual"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("Trajectory samples must establish path point lineage")
    try:
        sample_frames = _frame_array([s["source_frame"] for s in samples], "Sample source frames must be finite increasing integers")
    except (KeyError, TypeError) as error:
        raise ValueError("Every trajectory sample requires source-frame lineage") from error
    if (np.any(sample_frames < first) or np.any(sample_frames > last)
            or not set(frames.tolist()).issubset(set(sample_frames.tolist()))):
        raise ValueError("Path source-frame lineage is missing or exceeds episode bounds")
    by_frame = {float(s["source_frame"]): s for s in samples}
    for frame, point in zip(frames, xy):
        sample = by_frame[float(frame)]
        evidence = []
        if sample.get("relative_xy_m") is not None:
            evidence.append(np.asarray(sample["relative_xy_m"], dtype=float))
        if sample.get("center_px") is not None:
            pixel = np.asarray(sample["center_px"], dtype=float)
            bearing = np.asarray(record.get("bearing_center_px"), dtype=float)
            diameter = record.get("bearing_outer_diameter_px")
            if (pixel.shape != (2,) or bearing.shape != (2,) or not np.isfinite(pixel).all()
                    or not np.isfinite(bearing).all() or not isinstance(diameter, (int, float))
                    or not np.isfinite(diameter) or diameter <= 0):
                raise ValueError("Optical coordinate lineage requires finite bearing geometry and pixel XY")
            evidence.append((pixel-bearing)*np.array([1., -1.])*(.05/diameter))
        if origin == "manual_development_annotation" and not evidence:
            raise ValueError("A manual path point requires observed coordinate lineage")
        if any(value.shape != (2,) or not np.isfinite(value).all()
               or not np.allclose(value, point, rtol=1e-7, atol=1e-9) for value in evidence):
            raise ValueError("Derived path coordinate differs from its observed sample lineage")
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if not np.isfinite(arc).all() or arc[-1] <= 1e-8 or not np.allclose(progress, arc/arc[-1], rtol=1e-7, atol=1e-9):
        raise ValueError("Path arc progress is inconsistent with its finite XY points")
    displacement = xy[-1]-xy[0]
    length = float(np.hypot(*displacement))
    if not np.isfinite(length) or length < .005:
        raise ValueError("Video endpoint distance must be finite and at least 5mm")
    u = displacement/length
    basis = np.array([u, [-u[1], u[0]]])
    linear = (1-progress[:, None])*xy[0]+progress[:, None]*xy[-1]
    residual = xy-linear
    reference_angle = float(record.get("reference_angle", 0.))
    if not np.isfinite(reference_angle):
        raise ValueError("Trajectory reference angle must be finite")
    kinds = path.get("sample_kind", ["manual_visual" if origin == "manual_development_annotation" else "observed"]*len(progress))
    observed_kinds = {"observed", "manual", "manual_visual", "auto_reviewed"}
    if len(kinds) != len(progress) or any(k not in observed_kinds | {"interpolated", "short_gap_interpolated"} for k in kinds):
        raise ValueError("Sample kind must identify observed or interpolated evidence for every point")
    if "sample_weights" in path:
        confidence = np.asarray(path["sample_weights"], float)
        if confidence.shape != progress.shape or not np.isfinite(confidence).all() or np.any(confidence <= 0):
            raise ValueError("Optional sample evidence weights must be positive and finite")
    kinds = ["observed" if k in observed_kinds else "interpolated" for k in kinds]
    weights = np.r_[(progress[1]-progress[0])/2, (progress[2:]-progress[:-2])/2, (progress[-1]-progress[-2])/2]
    weights *= np.array([1. if k == "observed" else .25 for k in kinds])
    if weights.sum() <= 0:
        raise ValueError("Trajectory arc-cell weights require positive evidence")
    weights /= weights.sum()
    return dict(source=source, progress=progress, xy=xy, start=xy[0], goal=xy[-1],
        direction=reference_direction(displacement, reference_angle), basis=basis,
        local_residual_m=residual @ basis.T, residual_xy_m=residual, length_m=length,
        weights=weights, sample_kind=kinds, source_frames=frames.astype(int),
        reference_angle=reference_angle, origin=origin)


def prepare_development_samples(motion_payload, endpoint_payload, annotation_payload, *, seed=SEED):
    """Filter to original train/validation IDs before consuming any path labels."""
    source_sha = endpoint_payload.get("source", {}).get("sha256")
    annotation_sha = annotation_payload.get("source_sha256", annotation_payload.get("source", {}).get("sha256"))
    nested_annotation_sha = annotation_payload.get("source", {}).get("sha256", annotation_sha)
    if (not _valid_sha(source_sha) or motion_payload.get("source", {}).get("sha256") != source_sha
            or annotation_sha != source_sha or nested_annotation_sha != source_sha):
        raise ValueError("Motion, annotation and endpoint video provenance differ")
    if motion_payload.get("schema_version", 1) != 1 or annotation_payload.get("schema_version", 1) != 1:
        raise ValueError("Unsupported motion or annotation schema")
    episodes = endpoint_payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Endpoint manifest requires original source episodes")
    splits = split_episodes(episodes, seed)
    membership = {name: sorted(e["episode_id"] for e in group) for name, group in splits.items()}
    known = {e["episode_id"] for e in episodes}
    allowed = set(membership["train"]+membership["val"])
    old, manual = _records(motion_payload), _records(annotation_payload)
    if (set(old) | set(manual))-known:
        raise ValueError("Motion contains unknown source episode IDs")
    if set(manual)-allowed:
        raise ValueError("Development annotation source is outside training/validation membership")
    # Held-out old records are deliberately not validated or transformed here.
    # Their label values cannot affect a development fit or source normalizer.
    merged = {source: (manual[source], "manual_development_annotation") if source in manual
              else (old[source], "reviewed_v3_optical_track") for source in allowed if source in manual or source in old}
    prepared = {"train": [], "val": []}
    for name in prepared:
        for episode in sorted(splits[name], key=lambda e:e["episode_id"]):
            source = episode["episode_id"]
            if source not in merged:
                continue
            record, origin = merged[source]
            item = _prepare_record(record, episode, origin)
            if item is not None:
                prepared[name].append(item)
    if len(prepared["train"]) < 3:
        raise ValueError("Need at least three accepted independent training-source paths")
    return prepared, membership


def load_development_samples(real_motion, real_endpoints, annotations, *, seed=SEED):
    paths = [Path(value).resolve() for value in (real_motion, real_endpoints, annotations)]
    for path in paths:
        if (path.name.casefold() == "v4_holdout_annotations.json"
                or "v4_holdout_review" in {part.casefold() for part in path.parts}):
            raise ValueError("A training parser cannot consume a blind holdout artifact")
    motion_path, real_path, annotation_path = paths
    real = json.loads(real_path.read_text(encoding="utf-8"))
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    manual = json.loads(annotation_path.read_text(encoding="utf-8"))
    if motion.get("endpoint_manifest_sha256") != _hash(real_path):
        raise ValueError("Motion endpoint manifest provenance differs from exact saved bytes")
    source = real.get("source", {})
    basename = source.get("basename")
    if basename:
        if Path(basename).name != basename:
            raise ValueError("Video provenance basename must name one source file")
        video = real_path.parent/"source"/basename
        if not video.is_file() or _hash(video) != source.get("sha256"):
            raise ValueError("Source video file provenance differs from endpoint manifest")
    prepared, membership = prepare_development_samples(motion, real, manual, seed=seed)
    provenance = dict(original_video_sha256=source["sha256"], motion_sha256=_hash(motion_path),
        real_episodes_sha256=_hash(real_path), development_annotations_sha256=_hash(annotation_path),
        source_video_file_verified=bool(basename), input_sources={
            "real_motion": str(Path(real_motion)), "real_endpoints": str(Path(real_endpoints)),
            "development_annotations": str(Path(annotations))})
    return prepared, membership, provenance


def fit_direction_ridge(group, *, degree, alpha):
    if not group:
        raise ValueError("Cannot fit ridge without training-source paths")
    directions = np.array([item["direction"] for item in group])
    mean, std = directions.mean(0), np.maximum(directions.std(0), MIN_DIRECTION_STD)
    model = DirectionRidgePolicy(np.zeros((3*(degree+1), 2)), mean, std, degree=degree, alpha=alpha)
    x = np.concatenate([model.design_matrix(g["progress"], g["direction"]) for g in group])
    y = np.concatenate([g["local_residual_m"] for g in group])
    weights = np.concatenate([g["weights"]/len(group) for g in group])
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Ridge fitting requires finite weighted physical labels")
    coefficients = np.linalg.solve(x.T@(weights[:, None]*x)+alpha*np.eye(x.shape[1]), x.T@(weights[:, None]*y))
    return DirectionRidgePolicy(coefficients, mean, std, degree=degree, alpha=alpha)


def _source_metric(model, item):
    predicted = model.local_residuals(item["progress"], item["direction"])
    errors = np.linalg.norm(predicted-item["local_residual_m"], axis=1)
    baseline = np.linalg.norm(item["local_residual_m"], axis=1)
    return dict(source=item["source"], points=len(errors), observed_points=item["sample_kind"].count("observed") if "sample_kind" in item else len(errors),
        model_mean_displacement_m=float(errors @ item["weights"]), model_max_displacement_m=float(errors.max()),
        straight_mean_displacement_m=float(baseline @ item["weights"]), straight_max_displacement_m=float(baseline.max()))


def trajectory_metrics(model, group):
    per_source = [_source_metric(model, item) for item in group]
    return dict(sources=len(per_source), model_mean_displacement_m=float(np.mean([r["model_mean_displacement_m"] for r in per_source])) if per_source else None,
        straight_mean_displacement_m=float(np.mean([r["straight_mean_displacement_m"] for r in per_source])) if per_source else None, per_source=per_source)


def select_direction_ridge(train_group):
    """Locked six-candidate source-LOPO selection; validation never enters."""
    if len(train_group) < 3 or len({g["source"] for g in train_group}) != len(train_group):
        raise ValueError("Source-LOPO requires three or more unique training paths")
    candidates = []
    for degree, alpha in CANDIDATE_GRID:
        folds = []
        for heldout in train_group:
            fit_group = [item for item in train_group if item["source"] != heldout["source"]]
            model = fit_direction_ridge(fit_group, degree=degree, alpha=alpha)
            result = _source_metric(model, heldout)
            result.update(fit_sources=sorted(g["source"] for g in fit_group),
                direction_mean=model.direction_mean.tolist(), direction_std=model.direction_std.tolist())
            folds.append(result)
        candidates.append(dict(degree=degree, alpha=alpha, parameter_count=6*(degree+1),
            source_lopo_mean_displacement_m=float(np.mean([r["model_mean_displacement_m"] for r in folds])), folds=folds))
    best = min(candidates, key=lambda r:r["source_lopo_mean_displacement_m"])
    simple = [r for r in candidates if r["degree"] == 0 and r["source_lopo_mean_displacement_m"]-best["source_lopo_mean_displacement_m"] < SIMPLICITY_TOLERANCE_M]
    selected = min(simple, key=lambda r:r["source_lopo_mean_displacement_m"]) if simple else best
    model = fit_direction_ridge(train_group, degree=selected["degree"], alpha=selected["alpha"])
    selection = dict(method="training_source_leave_one_out", candidate_grid=[dict(degree=d, alpha=a) for d,a in CANDIDATE_GRID],
        candidates=candidates, selected_degree=selected["degree"], selected_alpha=selected["alpha"],
        best_source_lopo_mean_displacement_m=best["source_lopo_mean_displacement_m"],
        selected_source_lopo_mean_displacement_m=selected["source_lopo_mean_displacement_m"],
        simplicity_tolerance_m=SIMPLICITY_TOLERANCE_M,
        selection_rule="Lowest source-uniform LOO mean displacement; choose six coefficients when within strictly 0.1mm of best", validation_used_for_selection=False)
    return model, selection


def train_generalization(real_motion, real_endpoints, annotations, out):
    prepared, membership, provenance = load_development_samples(real_motion, real_endpoints, annotations)
    model, selection = select_direction_ridge(prepared["train"])
    metadata = dict(schema_version=1, method="real_video_direction_conditioned_ridge", seed=SEED,
        label_units="approximate image-plane metres", loss="source-uniform weighted physical squared XY displacement plus ridge penalty",
        representation="endpoint envelope 4p(1-p), degree zero/one, standardized reference-frame direction, local SI residual",
        reference_frame="camera/task XY; scene rotations require co-rotating reference_angle", parameter_count=model.parameter_count,
        maximum_residual_m=MAX_RESIDUAL_M, direction_std_floor=MIN_DIRECTION_STD,
        source_splits=membership, training_sources=sorted(g["source"] for g in prepared["train"]),
        validation_sources=sorted(g["source"] for g in prepared["val"]), training_points=sum(len(g["progress"]) for g in prepared["train"]),
        validation_points=sum(len(g["progress"]) for g in prepared["val"]),
        source_evidence=[dict(source=g["source"], split=split, origin=g["origin"], points=len(g["progress"]),
            observed_points=g["sample_kind"].count("observed"), interpolated_points=g["sample_kind"].count("interpolated"),
            source_frames=g["source_frames"].tolist()) for split, group in prepared.items() for g in group],
        selection=selection, environment=environment_signature(), implementation_sources=source_signature(), **provenance)
    output = Path(out)
    checkpoint = output/"policy.json"
    # Save the frozen policy before any validation diagnostics are calculated.
    save_generalization_policy(checkpoint, model, metadata)
    report = dict(metadata=metadata, frozen_checkpoint_sha256=_hash(checkpoint),
        trajectory_metrics={name:trajectory_metrics(model, group) for name,group in prepared.items()},
        metric_interpretation="Approximate optical XY displacement; validation diagnostic only; no test labels consumed")
    (output/"training.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default="data/real_motion.json")
    parser.add_argument("--real", default="data/real_episodes.json")
    parser.add_argument("--annotations", default="data/v4_development_annotations.json")
    parser.add_argument("--output", default="outputs/generalization_v4/model")
    args = parser.parse_args()
    model, metadata = train_generalization(args.motion, args.real, args.annotations, args.output)
    print(json.dumps(dict(parameter_count=model.parameter_count, degree=model.degree, alpha=model.alpha,
        training_sources=metadata["training_sources"], validation_sources=metadata["validation_sources"],
        selected_source_lopo_mean_displacement_m=metadata["selection"]["selected_source_lopo_mean_displacement_m"]), indent=2))


if __name__ == "__main__":
    main()
