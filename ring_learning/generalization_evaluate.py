"""Frozen procedural geometry and physical comparison of three path profiles.

Scene draws are independent procedural positions, not descendants of human
trajectories. The same V4 execution mechanism runs learned, straight and V3
profiles. Physical ring motion and command references are recorded separately.
This module never opens the blinded annotation labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import split_episodes
from .generalization_policy import load_generalization_policy, source_signature
from .generalization_train import CANDIDATE_GRID, SIMPLICITY_TOLERANCE_M, MIN_DIRECTION_STD, SEED
from .integrity import environment_signature, sha256_file


SCENE_SEED = 20261020
CONTROLLERS = ("v4", "straight", "v3_profile")
DISTANCE_BANDS_M = ((.045, .075), (.075, .110), (.110, .160))
ANGLE_BINS = 8
PLACEMENTS_PER_CELL = 4
EXPECTED_CASES = 96
DEFAULT_PROTOCOL = "docs/superpowers/specs/2026-10-14-generalization-design.md"
DEFAULT_BLIND_RECEIPT = "outputs/v4_holdout_review/freeze_receipt.json"
DEFAULT_V3_CHECKPOINT = "outputs/video_model/policy.pt"


def normalized_source_hash(module):
    path = Path(__file__).resolve().parent/f"{module}.py"
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
                          .replace("\r", "\n").encode("utf-8")).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")).hexdigest()


def training_protocol():
    return dict(method="direction_conditioned_local_metre_ridge",
        candidate_grid=[dict(degree=degree, alpha=alpha) for degree,alpha in CANDIDATE_GRID],
        selection="source_uniform_training_source_lopo_mae", fit_loss="source_uniform_weighted_physical_squared_error",
        simplicity_tolerance_m=SIMPLICITY_TOLERANCE_M, direction_std_floor=MIN_DIRECTION_STD,
        observed_weight=1., interpolated_weight=.25, maximum_residual_m=.025,
        source_split_seed=SEED, validation_used_for_selection=False, test_used_for_selection=False,
        progress_definition="normalized cumulative image-plane XY arc length")


def locked_selection_protocol_sha256():
    return _canonical_hash(training_protocol())


def _write_new_json(destination, value):
    destination = Path(destination)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _write_json(destination, value):
    Path(destination).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _label_path(path):
    return Path(path).as_posix()


def prepare(development, destination, *, real_path="data/real_episodes.json",
            protocol_path=DEFAULT_PROTOCOL, seed=SCENE_SEED):
    """Freeze the 96 predeclared strata without observing rollout outcomes."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("V4 scenes are frozen; choose a new destination")
    if seed != SCENE_SEED or isinstance(seed, bool):
        raise ValueError("The predeclared V4 scene seed is 20261020")
    real = json.loads(Path(real_path).read_text(encoding="utf-8"))
    annotation = json.loads(Path(development).read_text(encoding="utf-8"))
    source_sha = real["source"]["sha256"]
    if (annotation.get("source_sha256", annotation.get("source", {}).get("sha256")) != source_sha
            or annotation.get("source", {}).get("sha256", source_sha) != source_sha):
        raise ValueError("Development annotations and endpoint video provenance differ")
    real_motion = Path(real_path).parent/"real_motion.json"
    rng = np.random.default_rng(seed)
    configurations = []
    for distance_band, bounds in enumerate(DISTANCE_BANDS_M):
        for direction_bin in range(ANGLE_BINS):
            angle_lower = -np.pi+direction_bin*2*np.pi/ANGLE_BINS
            angle_upper = angle_lower+2*np.pi/ANGLE_BINS
            for position in range(PLACEMENTS_PER_CELL):
                distance = float(rng.uniform(*bounds))
                angle = float(rng.uniform(angle_lower, angle_upper))
                delta = distance*np.array([np.cos(angle), np.sin(angle)])
                # ring = bearing - delta. These conditional intervals are a
                # predeclared reset-workspace constraint, never a success filter.
                lower = np.maximum([.40, -.10], np.array([.32, -.23])+delta)
                upper = np.minimum([.52, .10], np.array([.62, .23])+delta)
                if np.any(lower > upper):
                    raise ValueError("Predeclared workspace has no feasible bearing interval")
                bearing = rng.uniform(lower, upper)
                ring = bearing-delta
                configurations.append(dict(configuration_id=f"G4_d{distance_band}_a{direction_bin}_p{position}",
                    split="procedural_test", distance_band_index=distance_band, direction_bin_index=direction_bin,
                    position_index=position, distance_band_m=list(bounds), endpoint_distance_m=distance,
                    angle_rad=angle, bearing_minus_ring_xy_m=delta.tolist(), ring_xy_m=ring.tolist(),
                    bearing_xy_m=bearing.tolist(), conditional_bearing_interval_m=dict(lower=lower.tolist(), upper=upper.tolist()),
                    reference_angle=0., geometry_scale=1., reset_seed=int(rng.integers(0, 2**31-1)),
                    ancestry="procedural geometry; no human source trajectory ancestor"))
    controller_hash = normalized_source_hash("generalization_control")
    manifest = dict(schema_version=1, protocol="generalization_v4_fixed_geometry_96", seed=seed,
        configurations=configurations, expected_cases=EXPECTED_CASES,
        environment=environment_signature(), controller_source_hash=controller_hash,
        execution_signature=dict(environment=environment_signature(), generalization_control_sha256=controller_hash,
            reference_max_step_m=.002, tool_lag_limit_m=.004, reference_angle=0., shared_across_controllers=list(CONTROLLERS)),
        training_protocol=training_protocol(), locked_selection_protocol_sha256=locked_selection_protocol_sha256(),
        training_protocol_path=_label_path(protocol_path), training_protocol_sha256=sha256_file(protocol_path),
        training_implementation_sha256=normalized_source_hash("generalization_train"),
        development_annotations_path=_label_path(development), development_annotations_sha256=sha256_file(development),
        real_episodes_path=_label_path(real_path), real_episodes_sha256=sha256_file(real_path),
        real_motion_path=_label_path(real_motion), real_motion_sha256=sha256_file(real_motion),
        source_video_sha256=source_sha,
        source_splits={name:sorted(e["episode_id"] for e in group) for name,group in split_episodes(real["episodes"], SEED).items()},
        dimensions_m=real["task_spec"]["dimensions_m"],
        geometry_protocol=dict(distance_bands_m=[list(b) for b in DISTANCE_BANDS_M], angle_bins=ANGLE_BINS,
            placements_per_cell=PLACEMENTS_PER_CELL, bearing_bounds_m=dict(lower=[.40,-.10],upper=[.52,.10]),
            ring_bounds_m=dict(lower=[.32,-.23],upper=[.62,.23]), delta_definition="bearing minus ring",
            sampling="uniform distance and angle within each stratum; uniform bearing within conditional feasible interval"),
        denominator_policy="All 96 predeclared cases for each controller; failures and execution exceptions retained",
        scope="Same objects, task and fixed reference; procedural position/direction/distance test, no independent human ancestry",
        outcome_filtering=False, frozen_before_selected_model=True)
    validate_manifest(manifest, development)
    _write_new_json(destination, manifest)
    return manifest


def validate_manifest(manifest, development=None):
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != "generalization_v4_fixed_geometry_96":
        raise ValueError("Unsupported frozen V4 scene protocol")
    if manifest.get("environment") != environment_signature():
        raise ValueError("Frozen scenes belong to a different simulation environment")
    controller_hash = normalized_source_hash("generalization_control")
    if (manifest.get("controller_source_hash") != controller_hash
            or manifest.get("execution_signature", {}).get("generalization_control_sha256") != controller_hash):
        raise ValueError("Frozen scenes belong to a different controller source")
    if manifest.get("training_implementation_sha256") != normalized_source_hash("generalization_train"):
        raise ValueError("Frozen scenes belong to a different training implementation")
    if (manifest.get("training_protocol") != training_protocol()
            or manifest.get("locked_selection_protocol_sha256") != locked_selection_protocol_sha256()
            or sha256_file(manifest["training_protocol_path"]) != manifest.get("training_protocol_sha256")):
        raise ValueError("Frozen training selection protocol differs")
    if sha256_file(development or manifest["development_annotations_path"]) != manifest.get("development_annotations_sha256"):
        raise ValueError("Frozen development annotation hash differs")
    if sha256_file(manifest["real_episodes_path"]) != manifest.get("real_episodes_sha256"):
        raise ValueError("Frozen endpoint manifest hash differs")
    if sha256_file(manifest["real_motion_path"]) != manifest.get("real_motion_sha256"):
        raise ValueError("Frozen old optical-motion hash differs")
    cases = manifest.get("configurations")
    if (not isinstance(cases, list) or len(cases) != EXPECTED_CASES or manifest.get("expected_cases") != EXPECTED_CASES
            or manifest.get("seed") != SCENE_SEED or manifest.get("outcome_filtering") is not False):
        raise ValueError("Frozen procedural evaluation must contain all 96 predeclared cases")
    identifiers, strata = set(), set()
    for case in cases:
        identifier = case.get("configuration_id")
        cell = (case.get("distance_band_index"),case.get("direction_bin_index"),case.get("position_index"))
        if (not isinstance(identifier, str) or identifier in identifiers or cell in strata
                or any(isinstance(i,bool) or not isinstance(i,int) for i in cell)
                or cell[0] not in range(3) or cell[1] not in range(8) or cell[2] not in range(4)
                or "source_episode_id" in case or "episode_id" in case):
            raise ValueError("Procedural scene identity or stratum membership is invalid")
        identifiers.add(identifier)
        strata.add(cell)
        ring, bearing = np.asarray(case["ring_xy_m"],float),np.asarray(case["bearing_xy_m"],float)
        if (ring.shape != (2,) or bearing.shape != (2,) or not np.isfinite(np.r_[ring,bearing]).all()
                or np.any(ring < np.array([.32,-.23])-1e-12) or np.any(ring > np.array([.62,.23])+1e-12)
                or np.any(bearing < np.array([.40,-.10])-1e-12) or np.any(bearing > np.array([.52,.10])+1e-12)):
            raise ValueError("Procedural geometry exceeds the predeclared finite workspace")
        delta = bearing-ring
        distance = float(np.hypot(*delta))
        lower, upper = DISTANCE_BANDS_M[cell[0]]
        angle = float(np.arctan2(delta[1],delta[0]))
        angle_lower = -np.pi+cell[1]*2*np.pi/ANGLE_BINS
        angle_upper = angle_lower+2*np.pi/ANGLE_BINS
        if (not lower-1e-12 <= distance <= upper+1e-12 or not angle_lower-1e-12 <= angle <= angle_upper+1e-12
                or not np.allclose(delta,case["bearing_minus_ring_xy_m"],atol=1e-12,rtol=0)
                or not np.isclose(distance,case["endpoint_distance_m"],atol=1e-12,rtol=0)
                or case.get("reference_angle") != 0. or case.get("geometry_scale") != 1.):
            raise ValueError("Procedural distance, direction or reference frame differs from its stratum")
    return True


def _validate_model_binding(metadata, manifest):
    for model_key, manifest_key in (("development_annotations_sha256","development_annotations_sha256"),
            ("real_episodes_sha256","real_episodes_sha256"),("motion_sha256","real_motion_sha256"),
            ("original_video_sha256","source_video_sha256")):
        if metadata.get(model_key) != manifest.get(manifest_key):
            raise ValueError("Selected checkpoint data provenance differs from frozen scenes")
    training = set(metadata.get("training_sources", []))
    validation = set(metadata.get("validation_sources", []))
    if (not training or not training.issubset(manifest["source_splits"]["train"])
            or not validation.issubset(manifest["source_splits"]["val"])):
        raise ValueError("Selected checkpoint source membership differs from frozen development splits")
    selection = metadata.get("selection", {})
    if (selection.get("candidate_grid") != training_protocol()["candidate_grid"]
            or selection.get("validation_used_for_selection") is not False):
        raise ValueError("Selected checkpoint does not use the locked training-source selection protocol")


def freeze_model_receipt(manifest, checkpoint, development, receipt, *,
                         blind_receipt=DEFAULT_BLIND_RECEIPT, v3_checkpoint=DEFAULT_V3_CHECKPOINT):
    """Bind selected model to metadata-only blind receipt before unsealing labels."""
    receipt = Path(receipt)
    if receipt.exists():
        raise FileExistsError("Model/protocol freeze receipt is immutable")
    manifest_path = Path(manifest)
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(frozen, development)
    _, metadata = load_generalization_policy(checkpoint)
    _validate_model_binding(metadata, frozen)
    # Only this declared receipt is opened. Its annotation_path is never read.
    blind = json.loads(Path(blind_receipt).read_text(encoding="utf-8"))
    sources = sorted(blind.get("reserved_source_ids", []))
    annotation_sha = blind.get("annotation_sha256")
    if (sources != ["E20","E21","E22"] or not set(sources).issubset(frozen["source_splits"]["ood_test"])
            or set(sources) & set(metadata["training_sources"])
            or set(sources) & set(metadata["validation_sources"])):
        raise ValueError("Blind holdout reserved source IDs or source isolation differ")
    if (not isinstance(annotation_sha,str) or len(annotation_sha) != 64
            or any(c not in "0123456789abcdef" for c in annotation_sha)
            or blind.get("source_sha256") != frozen["source_video_sha256"]
            or blind.get("endpoint_manifest_sha256") != frozen["real_episodes_sha256"]
            or blind.get("existing_motion_sha256") != frozen["real_motion_sha256"]):
        raise ValueError("Blind holdout receipt provenance differs from frozen development data")
    result = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(), frozen_before_holdout_unseal=True,
        selected_checkpoint_path=_label_path(checkpoint), selected_checkpoint_sha256=sha256_file(checkpoint),
        training_protocol_path=frozen["training_protocol_path"], training_protocol_sha256=frozen["training_protocol_sha256"],
        locked_selection_protocol_sha256=frozen["locked_selection_protocol_sha256"],
        development_annotations_path=_label_path(development), development_annotations_sha256=sha256_file(development),
        holdout_annotations_sha256=annotation_sha, holdout_sources=sources,
        blind_receipt_path=_label_path(blind_receipt), blind_receipt_sha256=sha256_file(blind_receipt),
        holdout_valid_episodes=blind.get("valid_episodes"), holdout_measured_samples=blind.get("measured_samples"),
        real_episodes_path=frozen["real_episodes_path"], real_episodes_sha256=frozen["real_episodes_sha256"],
        original_video_sha256=frozen["source_video_sha256"], training_sources=metadata["training_sources"],
        validation_sources=metadata["validation_sources"], manifest_path=_label_path(manifest), manifest_sha256=sha256_file(manifest),
        controller_source_hash=frozen["controller_source_hash"], policy_train_sources=source_signature(),
        evaluator_source_hash=normalized_source_hash("generalization_evaluate"),
        v3_checkpoint_path=_label_path(v3_checkpoint), v3_checkpoint_sha256=sha256_file(v3_checkpoint),
        environment=frozen["environment"], execution_signature=frozen["execution_signature"])
    _write_new_json(receipt, result)
    return result


def _snapshot(env, controller, step, phase_before, action=None):
    ring, command, tool = [np.asarray(getattr(env,name),float) for name in ("ring_xyz","command_xyz","tool_xyz")]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (ring,command,tool)):
        raise ValueError("Physical rollout state must be finite XYZ")
    return dict(step=step, phase_before=int(phase_before), phase_after=int(env.phase),
        ring_xyz_m=ring.tolist(), command_xyz_m=command.tolist(), tool_xyz_m=tool.tolist(),
        finger_contacts=int(env.ring_finger_contacts()),
        progress=float(controller.progress), arc_cursor_m=float(controller.arc_cursor_m),
        action=None if action is None else action.tolist(), tool_command_lag_m=float(np.linalg.norm(tool-command)))


def run_physical_case(env, configuration, controller, name):
    """Attempt one reset/rollout; exceptions are recorded as failures, no rescue."""
    trajectory, transfer, actions, info = [], [], [], {}
    steps, success = 0, False
    try:
        observation = env.reset(configuration["ring_xy_m"],configuration["bearing_xy_m"],configuration["reset_seed"])
        trajectory.append(_snapshot(env,controller,0,env.phase))
        for step in range(env.max_steps):
            phase_before = env.phase
            if phase_before == 4 and not transfer:
                transfer.append(_snapshot(env,controller,steps,phase_before))
            action = np.asarray(controller(env,observation),float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("Physical controller must return a finite four-component action")
            action = np.clip(action,-1.,1.)
            observation, done, info = env.step(action)
            steps = step+1
            sample = _snapshot(env,controller,steps,phase_before,action)
            trajectory.append(sample)
            actions.append(action.tolist())
            if phase_before == 4:
                transfer.append(sample)
            if done:
                break
        success = bool(env.success())
        if not success and not info.get("failure_reason"):
            info = dict(info, failure_reason="max_steps_without_success")
    except Exception as error:
        info = dict(info, failure_reason="evaluation_exception", exception=f"{type(error).__name__}: {error}")
        success = False
    transfer_steps = transfer[1:]
    lags = [sample["tool_command_lag_m"] for sample in transfer_steps]
    counters = {key:int(getattr(controller,key,0)) for key in
        ("learned_reference_evaluations","transfer_steps","planned_steps","lag_pause_steps")}
    return dict(controller=name, configuration_id=configuration["configuration_id"], split="procedural_test",
        distance_band_index=configuration["distance_band_index"], direction_bin_index=configuration["direction_bin_index"],
        position_index=configuration["position_index"], attempted=True, success=success, steps=steps, info=info,
        trajectory=trajectory, executed_actions=actions,
        transfer_ring_xy_m=[s["ring_xyz_m"][:2] for s in transfer],
        transfer_command_xy_m=[s["command_xyz_m"][:2] for s in transfer],
        transfer_tool_xy_m=[s["tool_xyz_m"][:2] for s in transfer],
        transfer_reference_progress=[s["progress"] for s in transfer],
        transfer_contact_steps=sum(s["finger_contacts"]>0 for s in transfer_steps),
        maximum_finger_contacts=max([s["finger_contacts"] for s in trajectory],default=0),
        transfer_tool_command_mean_lag_m=float(np.mean(lags)) if lags else None,
        transfer_tool_command_max_lag_m=float(np.max(lags)) if lags else None,
        reference_curve_length_m=float(getattr(controller,"curve_length_m",0.)),
        reference_final_progress=float(getattr(controller,"progress",0.)),
        reference_final_arc_cursor_m=float(getattr(controller,"arc_cursor_m",0.)), **counters)


def _summary(rows, expected):
    failures = {}
    for row in rows:
        if not row["success"]:
            reason = row["info"].get("failure_reason") or "unknown"
            failures[reason] = failures.get(reason,0)+1
    missing = len(expected)-len(rows)
    if missing:
        failures["not_executed"] = missing
    successes = sum(bool(r["success"]) for r in rows)
    lag_rows = [r for r in rows if r.get("transfer_tool_command_mean_lag_m") is not None]
    return dict(episodes=len(expected), attempted_cases=len(rows), successes=successes,
        success_rate=successes/len(expected) if expected else None, failure_counts=failures,
        mean_steps_per_attempt=float(np.mean([r["steps"] for r in rows])) if rows else None,
        learned_reference_evaluations=sum(r.get("learned_reference_evaluations",0) for r in rows),
        transfer_steps=sum(r.get("transfer_steps",0) for r in rows), planned_steps=sum(r.get("planned_steps",0) for r in rows),
        lag_pause_steps=sum(r.get("lag_pause_steps",0) for r in rows), transfer_contact_steps=sum(r.get("transfer_contact_steps",0) for r in rows),
        transfer_tool_command_source_mean_lag_m=float(np.mean([r["transfer_tool_command_mean_lag_m"] for r in lag_rows])) if lag_rows else None,
        transfer_tool_command_max_lag_m=max([r["transfer_tool_command_max_lag_m"] for r in lag_rows],default=None))


def summarize(rows, configurations):
    known = {c["configuration_id"] for c in configurations}
    pairs = [(r["controller"],r["configuration_id"]) for r in rows]
    if len(pairs) != len(set(pairs)) or any(name not in CONTROLLERS or case not in known for name,case in pairs):
        raise ValueError("Evaluation rows must have unique known controller/configuration identities")
    overall, distances, directions, cells = {}, {}, {}, {}
    for name in CONTROLLERS:
        group = [r for r in rows if r["controller"] == name]
        overall[name] = _summary(group,configurations)
        distances[name] = {str(i):_summary([r for r in group if r["distance_band_index"]==i],
            [c for c in configurations if c["distance_band_index"]==i]) for i in range(3)}
        directions[name] = {str(i):_summary([r for r in group if r["direction_bin_index"]==i],
            [c for c in configurations if c["direction_bin_index"]==i]) for i in range(8)}
        cells[name] = {f"{d}/{a}":_summary([r for r in group if r["distance_band_index"]==d and r["direction_bin_index"]==a],
            [c for c in configurations if c["distance_band_index"]==d and c["direction_bin_index"]==a]) for d in range(3) for a in range(8)}
    return dict(overall=overall,by_distance_band=distances,by_direction_bin=directions,by_distance_direction_cell=cells)


def evaluate(manifest_path, checkpoint, output, *, v3_checkpoint=DEFAULT_V3_CHECKPOINT,
             receipt="outputs/generalization_v4/model_freeze.json"):
    from .generalization_control import GeneralizationController, GeneralizationEnv, LegacyProfileAdapter
    from .video_policy import load_video_policy
    from .sim import CONTROL_DT
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    model, metadata = load_generalization_policy(checkpoint)
    _validate_model_binding(metadata,manifest)
    frozen = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if (frozen.get("selected_checkpoint_sha256") != sha256_file(checkpoint)
            or frozen.get("manifest_sha256") != sha256_file(manifest_path)
            or frozen.get("v3_checkpoint_sha256") != sha256_file(v3_checkpoint)
            or frozen.get("evaluator_source_hash") != normalized_source_hash("generalization_evaluate")
            or frozen.get("controller_source_hash") != normalized_source_hash("generalization_control")):
        raise ValueError("Physical evaluation sources differ from the model/protocol freeze receipt")
    legacy, _ = load_video_policy(v3_checkpoint)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    env, rows = GeneralizationEnv(), []
    try:
        for name in CONTROLLERS:
            for index,cfg in enumerate(manifest["configurations"]):
                profile = LegacyProfileAdapter(legacy) if name == "v3_profile" else model
                controller = GeneralizationController(profile, geometry=name=="straight", reference_angle=0.)
                row = run_physical_case(env,cfg,controller,name)
                rows.append(row)
                if (index+1)%8 == 0 or index+1 == EXPECTED_CASES:
                    successful = sum(r["success"] for r in rows if r["controller"]==name)
                    print(f"{name} {index+1}/{EXPECTED_CASES}: successes={successful}",flush=True)
                    _write_json(output/"rollouts.json",rows)
    finally:
        env.close()
        _write_json(output/"rollouts.json",rows)
    report = dict(schema_version=1,protocol=manifest["protocol"],manifest_sha256=sha256_file(manifest_path),
        selected_checkpoint_sha256=sha256_file(checkpoint), v3_checkpoint_sha256=sha256_file(v3_checkpoint),
        freeze_receipt_sha256=sha256_file(receipt), summaries=summarize(rows,manifest["configurations"]),
        rows=len(rows), expected_rows=EXPECTED_CASES*len(CONTROLLERS), training_metadata=metadata,
        environment=manifest["environment"], execution_signature=manifest["execution_signature"], control_dt_s=CONTROL_DT,
        controller_scope="Shared V4 curve arc/lag execution and reference-complete transfer gate; geometric remaining phases, Z and grip; original IK/contact physics",
        trajectory_definition="Saved initial and post-step physical ring/tool states; command reference is separate; actions are executed clipped actions",
        counter_definition="learned_reference_evaluations counts true profile waypoint calls while building transfer curve; straight has zero",
        all_failures_retained=True,human_test_labels_used=False,scope=manifest["scope"])
    _write_json(output/"results.json",report)
    print(json.dumps(report["summaries"]["overall"],indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--freeze-model",action="store_true")
    parser.add_argument("--manifest",default="outputs/generalization_v4/configurations.json")
    parser.add_argument("--annotations",default="data/v4_development_annotations.json")
    parser.add_argument("--protocol",default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed",type=int,default=SCENE_SEED)
    parser.add_argument("--receipt",default="outputs/generalization_v4/model_freeze.json")
    parser.add_argument("--checkpoint",default="outputs/generalization_v4/model/policy.json")
    parser.add_argument("--real",default="data/real_episodes.json")
    parser.add_argument("--v3-checkpoint",default=DEFAULT_V3_CHECKPOINT)
    parser.add_argument("--blind-receipt",default=DEFAULT_BLIND_RECEIPT)
    parser.add_argument("--output",default="outputs/generalization_v4/evaluation")
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare(args.annotations,args.manifest,real_path=args.real,protocol_path=args.protocol,seed=args.seed)
        print(json.dumps(dict(cases=len(manifest["configurations"]),manifest_sha256=sha256_file(args.manifest),
            development_annotations_sha256=manifest["development_annotations_sha256"],controller_source_hash=manifest["controller_source_hash"]),indent=2))
    elif args.freeze_model:
        print(json.dumps(freeze_model_receipt(args.manifest,args.checkpoint,args.annotations,args.receipt,
            blind_receipt=args.blind_receipt,v3_checkpoint=args.v3_checkpoint),indent=2))
    else:
        evaluate(args.manifest,args.checkpoint,args.output,v3_checkpoint=args.v3_checkpoint,receipt=args.receipt)


if __name__ == "__main__":
    main()
