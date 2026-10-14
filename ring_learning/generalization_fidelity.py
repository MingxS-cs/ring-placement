"""Frozen optical path fidelity at observed source arc progress, never human timing.

This evaluation opens the locked labels only after the model, development data,
selection protocol, comparison checkpoint, and blind artifact hashes are checked.
Its synthetic tests do not access the real blind annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import split_episodes
from .generalization_policy import load_generalization_policy, source_signature
from .generalization_train import SEED, _prepare_record, _records, _valid_sha
from .video_policy import load_video_policy


HOLDOUT_SOURCES = ("E20", "E21", "E22")
OBSERVED_POINTS = 35
INTERMEDIATE_POINTS = 29
SENSITIVITY_SEED = 20261021
SENSITIVITY_DRAWS = 100
SENSITIVITY_RADII_PX = (5, 8)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hash(path):
    encoded = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_file(path, expected, description):
    path = Path(path).resolve()
    if not _valid_sha(expected) or not path.is_file() or _hash(path) != expected:
        raise ValueError(f"Frozen {description} file hash/provenance differs from model freeze receipt")
    return path


def _receipt_path(receipt, key):
    value = receipt.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Model freeze receipt requires {key}")
    return Path(value).resolve()


def _validate_freeze(checkpoint, receipt_path, holdout, v3_checkpoint):
    """All pre-unseal checks; hashing blind bytes does not parse label content."""
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version", 1) != 1:
        raise ValueError("Unsupported model freeze receipt schema")
    checkpoint = _frozen_file(checkpoint, receipt.get("selected_checkpoint_sha256"), "selected checkpoint")
    model, metadata = load_generalization_policy(checkpoint)
    protocol = _frozen_file(_receipt_path(receipt, "training_protocol_path"), receipt.get("training_protocol_sha256"), "training protocol")
    development = _frozen_file(_receipt_path(receipt, "development_annotations_path"), receipt.get("development_annotations_sha256"), "development annotations")
    endpoints = _frozen_file(_receipt_path(receipt, "real_episodes_path"), receipt.get("real_episodes_sha256"), "original endpoint manifest")
    for key in ("development_annotations_sha256", "real_episodes_sha256", "original_video_sha256"):
        if metadata.get(key) != receipt.get(key) or not _valid_sha(receipt.get(key)):
            raise ValueError(f"Selected checkpoint {key} provenance differs from model freeze receipt")
    if metadata.get("implementation_sources") != source_signature():
        raise ValueError("Selected checkpoint policy/training implementation provenance is not frozen")
    if "policy_train_sources" in receipt and receipt["policy_train_sources"] != source_signature():
        raise ValueError("Frozen policy/training source hashes differ from current implementation")
    selection = metadata.get("selection", {})
    if (metadata.get("method") != "real_video_direction_conditioned_ridge" or metadata.get("seed") != SEED
            or selection.get("method") != "training_source_leave_one_out"
            or selection.get("validation_used_for_selection") is not False
            or selection.get("selected_degree") != model.degree
            or selection.get("selected_alpha") != model.alpha):
        raise ValueError("Selected model does not match frozen training/selection protocol")
    if "locked_selection_protocol_sha256" in receipt:
        from .generalization_evaluate import locked_selection_protocol_sha256
        if receipt["locked_selection_protocol_sha256"] != locked_selection_protocol_sha256():
            raise ValueError("Frozen locked selection protocol hash differs from current protocol")
    for path_key, hash_key, description in (
            ("manifest_path", "manifest_sha256", "physical scene manifest"),
            ("blind_receipt_path", "blind_receipt_sha256", "blind annotation receipt")):
        if path_key in receipt or hash_key in receipt:
            _frozen_file(_receipt_path(receipt, path_key), receipt.get(hash_key), description)
    for key, filename in (("controller_source_hash", "generalization_control.py"),
                          ("evaluator_source_hash", "generalization_evaluate.py")):
        if key in receipt and receipt[key] != _source_hash(Path(__file__).with_name(filename)):
            raise ValueError(f"Frozen {key} differs from current source implementation")
    original = json.loads(endpoints.read_text(encoding="utf-8"))
    source = original.get("source", {})
    if source.get("sha256") != receipt["original_video_sha256"]:
        raise ValueError("Original video provenance differs from model freeze receipt")
    basename = source.get("basename")
    if not isinstance(basename, str) or Path(basename).name != basename:
        raise ValueError("Original source video provenance needs a single-file basename")
    video = _frozen_file(endpoints.parent/"source"/basename, source["sha256"], "original source video")
    episodes = original.get("episodes", [])
    by_id = {e["episode_id"]:e for e in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("Original endpoint manifest contains duplicate source episodes")
    membership = {name:sorted(e["episode_id"] for e in group) for name,group in split_episodes(episodes, SEED).items()}
    training = metadata.get("training_sources")
    if (not isinstance(training, list) or len(training) != 13 or len(set(training)) != 13
            or sorted(training) != membership["train"] or sorted(training) != sorted(receipt.get("training_sources", []))
            or set(training) & set(HOLDOUT_SOURCES)):
        raise ValueError("Frozen training source membership overlaps holdout or differs from original thirteen sources")
    if metadata.get("source_splits") != membership:
        raise ValueError("Checkpoint source split membership differs from original frozen episode membership")
    if (sorted(receipt.get("holdout_sources", [])) != list(HOLDOUT_SOURCES)
            or not set(HOLDOUT_SOURCES).issubset(membership["ood_test"])):
        raise ValueError("Frozen holdout sources must be the original reserved E20/E21/E22 episodes")
    legacy_path = _receipt_path(receipt, "v3_checkpoint_path") if v3_checkpoint is None else Path(v3_checkpoint).resolve()
    legacy_path = _frozen_file(legacy_path, receipt.get("v3_checkpoint_sha256"), "V3 comparison checkpoint")
    legacy, legacy_metadata = load_video_policy(legacy_path)
    if set(legacy_metadata.get("training_sources", [])) & set(HOLDOUT_SOURCES):
        raise ValueError("V3 comparison training sources overlap the fresh reserved holdout")
    if ("original_video_sha256" in legacy_metadata
            and legacy_metadata["original_video_sha256"] != receipt["original_video_sha256"]):
        raise ValueError("V3 comparison source video provenance differs from frozen source")
    # This is deliberately the last file hash before blind JSON parsing below.
    holdout = _frozen_file(holdout, receipt.get("holdout_annotations_sha256"), "holdout annotations")
    provenance = dict(model_freeze_receipt_sha256=_hash(receipt_path), selected_checkpoint_sha256=_hash(checkpoint),
        training_protocol_sha256=_hash(protocol), development_annotations_sha256=_hash(development),
        real_episodes_sha256=_hash(endpoints), original_video_sha256=_hash(video),
        holdout_annotations_sha256=_hash(holdout), v3_checkpoint_sha256=_hash(legacy_path),
        fidelity_source_hash=_source_hash(__file__), training_sources=sorted(training))
    return model, legacy, receipt, holdout, by_id, provenance


def _prepare_holdout(payload, original_episodes, source_sha):
    """Blind evaluation preparation, never the development/training parser."""
    if payload.get("schema_version", 1) != 1:
        raise ValueError("Unsupported holdout annotation schema")
    declared = payload.get("source_sha256", payload.get("source", {}).get("sha256"))
    if (declared != source_sha or payload.get("source", {}).get("sha256", declared) != source_sha):
        raise ValueError("Holdout annotation source video SHA provenance differs from frozen source")
    records = _records(payload)
    if sorted(records) != list(HOLDOUT_SOURCES):
        raise ValueError("Blind evaluation requires only original reserved E20/E21/E22 sources")
    group = []
    observed_kinds = {"observed", "manual", "manual_visual", "auto_reviewed"}
    for source in HOLDOUT_SOURCES:
        record = records[source]
        prepared = _prepare_record(record, original_episodes[source], "manual_development_annotation")
        if prepared is None:
            raise ValueError("Every locked holdout source requires an accepted observed path")
        samples = {s["source_frame"]:s for s in record.get("samples", record.get("samples_manual_visual", []))}
        if (any(samples[int(f)].get("kind") not in observed_kinds for f in prepared["source_frames"])
                or any(k != "observed" for k in prepared["sample_kind"])):
            raise ValueError("Locked fidelity requires genuinely observed samples; interpolated points are excluded")
        if prepared["reference_angle"] != 0:
            raise ValueError("V3 fidelity comparison is defined in the fixed camera reference frame")
        group.append(prepared)
    count = sum(len(g["progress"]) for g in group)
    if count != OBSERVED_POINTS or count-2*len(group) != INTERMEDIATE_POINTS:
        raise ValueError("Locked holdout observed/intermediate counts differ from predeclared 35/29 points")
    return group, records


def _arc_weights(progress):
    p = np.asarray(progress, float)
    if (p.ndim != 1 or len(p) < 3 or not np.isfinite(p).all()
            or p[0] != 0 or p[-1] != 1 or np.any(np.diff(p) < 0)):
        raise ValueError("Observed arc progress must be finite, monotonic, and endpoint anchored")
    weights = np.r_[(p[1]-p[0])/2, (p[2:]-p[:-2])/2, (p[-1]-p[-2])/2]
    if weights.sum() <= 0 or weights[1:-1].sum() <= 0:
        raise ValueError("Observed paths need positive intermediate arc-cell weight")
    return weights/weights.sum()


def _error_metrics(errors, weights):
    interior = weights[1:-1]/weights[1:-1].sum()
    return dict(mean_xy_error_m=float(errors @ weights), rms_xy_error_m=float(np.sqrt(errors**2 @ weights)),
        maximum_xy_error_m=float(errors.max()), mean_intermediate_xy_error_m=float(errors[1:-1] @ interior),
        rms_intermediate_xy_error_m=float(np.sqrt(errors[1:-1]**2 @ interior)),
        maximum_intermediate_xy_error_m=float(errors[1:-1].max()), intermediate_arc_weights=interior.tolist())


def fidelity_metrics(model, legacy, group):
    """Compare all curves at exactly the same observed, geometric arc progress."""
    if not group or len({g["source"] for g in group}) != len(group):
        raise ValueError("Fidelity requires distinct original source episodes")
    rows = []
    for item in sorted(group, key=lambda g:g["source"]):
        p, xy, start, goal = (item[name] for name in ("progress", "xy", "start", "goal"))
        weights = _arc_weights(p)
        if np.asarray(xy).shape != (len(p),2) or not np.isfinite(xy).all():
            raise ValueError("Observed path requires finite XY for every progress point")
        predicted = dict(v4=np.asarray([model.waypoint(start, goal, float(q), reference_angle=item.get("reference_angle", 0.)) for q in p]),
            straight=(1-p[:,None])*start+p[:,None]*goal,
            v3=np.asarray([legacy.waypoint(start, goal, float(q)) for q in p]))
        if any(value.shape != xy.shape or not np.isfinite(value).all() for value in predicted.values()):
            raise ValueError("A comparison policy produced nonfinite or malformed XY predictions")
        errors = {name:np.linalg.norm(value-xy,axis=1) for name,value in predicted.items()}
        rows.append(dict(source_episode_id=item["source"],observed_points=len(p),intermediate_points=len(p)-2,
            source_frames=item["source_frames"].tolist(), observed_arc_progress=p.tolist(),
            observed_relative_xy_m=xy.tolist(), predicted_relative_xy_m={name:value.tolist() for name,value in predicted.items()},
            point_xy_errors_m={name:value.tolist() for name,value in errors.items()},arc_cell_weights=weights.tolist(),
            methods={name:_error_metrics(value,weights) for name,value in errors.items()}))
    summary = {}
    for name in ("v4","straight","v3"):
        metrics = [row["methods"][name] for row in rows]
        summary[name] = dict(source_uniform_mean_xy_error_m=float(np.mean([r["mean_xy_error_m"] for r in metrics])),
            source_uniform_rms_xy_error_m=float(np.sqrt(np.mean([r["rms_xy_error_m"]**2 for r in metrics]))),
            maximum_xy_error_m=max(r["maximum_xy_error_m"] for r in metrics),
            source_uniform_mean_intermediate_xy_error_m=float(np.mean([r["mean_intermediate_xy_error_m"] for r in metrics])),
            source_uniform_rms_intermediate_xy_error_m=float(np.sqrt(np.mean([r["rms_intermediate_xy_error_m"]**2 for r in metrics]))),
            maximum_intermediate_xy_error_m=max(r["maximum_intermediate_xy_error_m"] for r in metrics))
    return dict(sources=len(rows), source_episode_ids=[r["source_episode_id"] for r in rows],
        observed_points=sum(r["observed_points"] for r in rows),intermediate_points=sum(r["intermediate_points"] for r in rows),
        methods=summary,per_source=rows,progress_interpretation="Observed normalized image-plane XY arc length; not human action timing",
        weighting="Each source has equal weight. Whole-path XY errors use observed arc-cell widths; intermediate errors exclude both endpoints and renormalize interior arc-cell widths within each source.")


def perturb_observed_path(item, record, rng, radius_px):
    """Coordinate measurement perturbation; retain true endpoints as anchors."""
    diameter = record.get("bearing_outer_diameter_px")
    if not isinstance(diameter, (int,float)) or not np.isfinite(diameter) or diameter <= 0 or radius_px <= 0:
        raise ValueError("Measurement perturbation requires positive optical scale and radius")
    xy = np.array(item["xy"],float,copy=True)
    xy[1:-1] += rng.uniform(-radius_px,radius_px,(len(xy)-2,2))*np.array([1.,-1.])*(.05/diameter)
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if not np.isfinite(arc).all() or arc[-1] <= 0:
        raise ValueError("Perturbed observed path must retain finite positive arc length")
    progress = arc/arc[-1]
    return dict(item,xy=xy,start=xy[0],goal=xy[-1],progress=progress,weights=_arc_weights(progress))


def measurement_sensitivity(model, legacy, group, records):
    radii = []
    for radius in SENSITIVITY_RADII_PX:
        # The same predeclared unit-square draws are scaled for both radii.
        rng = np.random.default_rng(SENSITIVITY_SEED)
        draws = []
        for index in range(SENSITIVITY_DRAWS):
            perturbed = [perturb_observed_path(item,records[item["source"]],rng,radius) for item in sorted(group,key=lambda g:g["source"])]
            metrics = fidelity_metrics(model,legacy,perturbed)
            draws.append(dict(draw=index,methods=metrics["methods"]))
        summaries = {}
        for name in ("v4","straight","v3"):
            summaries[name] = {}
            for metric in ("source_uniform_mean_xy_error_m","source_uniform_mean_intermediate_xy_error_m"):
                values = np.array([d["methods"][name][metric] for d in draws])
                summaries[name][metric] = dict(mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
                    perturbation_percentile_5=float(np.quantile(values,.05)),perturbation_percentile_95=float(np.quantile(values,.95)))
        radii.append(dict(radius_px=radius,methods=summaries,draws=draws))
    return dict(seed=SENSITIVITY_SEED,draws_per_radius=SENSITIVITY_DRAWS,square_uniform_radius_px=list(SENSITIVITY_RADII_PX),
        endpoint_anchors_fixed=True,arc_progress_recomputed=True,paired_unit_square_draws_across_radii=True,
        interpretation="Measurement perturbation only; not a confidence interval, retraining, bootstrap of independent episodes, or human action timing. Intermediate optical center coordinates vary; true endpoints remain fixed.",
        radius_results=radii)


def evaluate_fidelity(checkpoint, model_freeze_receipt, holdout, output, *, v3_checkpoint=None):
    """Unseal only after frozen bindings pass; root must authorize the real run."""
    model,legacy,receipt,holdout,original,provenance = _validate_freeze(checkpoint,model_freeze_receipt,holdout,v3_checkpoint)
    # Do not move this read above any pre-unseal validation.
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    group,records = _prepare_holdout(payload,original,receipt["original_video_sha256"])
    report = fidelity_metrics(model,legacy,group)
    report.update(schema_version=1,provenance=provenance,model_selection_frozen_before_label_open=True,
        label_units="approximate image-plane metres",endpoint_anchor_disclosure="All compared curves use the true observed start/end XY as anchors; intermediate-only metrics remove exact endpoint agreement.",
        measurement_sensitivity=measurement_sensitivity(model,legacy,group,records),
        limitations="Three correlated reserved windows from the same fixed-camera recording, with prior endpoint exposure. Optical path fidelity does not establish calibrated 3D, grasp/force/depth/timing recovery, or new-camera/object/real-robot generalization.")
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination = destination/"fidelity.json"
    if destination.resolve() in {Path(checkpoint).resolve(),Path(model_freeze_receipt).resolve(),holdout}:
        raise ValueError("Fidelity output cannot overwrite a frozen input artifact")
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="outputs/generalization_v4/model/policy.json")
    parser.add_argument("--model-freeze-receipt",default="outputs/generalization_v4/model_freeze.json")
    parser.add_argument("--holdout",default="data/v4_holdout_annotations.json")
    parser.add_argument("--output",default="outputs/generalization_v4/fidelity")
    parser.add_argument("--v3-checkpoint")
    args = parser.parse_args()
    report = evaluate_fidelity(args.checkpoint,args.model_freeze_receipt,args.holdout,args.output,v3_checkpoint=args.v3_checkpoint)
    print(json.dumps(dict(sources=report["sources"],observed_points=report["observed_points"],
        intermediate_points=report["intermediate_points"],methods=report["methods"]),indent=2))


if __name__ == "__main__":
    main()
