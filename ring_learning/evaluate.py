"""Closed-loop robot evaluation; student receives no teacher actions or rescue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .dataset import noisy_observation, validate_lineage, write_json
from .generate import run_controller
from .policy import Policy, load_checkpoint
from .integrity import verify_binding, sha256_file


def validate_evaluation_request(configurations, splits, limit):
    if not splits or len(set(splits)) != len(splits) or any(s not in configurations for s in splits):
        raise ValueError("Evaluation splits must be nonempty, unique and known")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("Evaluation limit must be a positive integer")


def save_video(path, frames, fps=20):
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "21", str(path)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode rollout video")


def evaluate(dataset_dir, checkpoint, output_dir, splits=("test", "ood_test"), noise=0.,
             include_baselines=True, render_one=True, limit=None):
    from .sim import CONTROL_DT, RingEnv
    from .teacher import teacher

    torch.set_num_threads(2)
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir/"configurations.json").read_text(encoding="utf-8"))
    configurations = manifest["configurations"]
    validate_evaluation_request(configurations, splits, limit)
    validate_lineage(configurations)
    model, normalizer, metadata = load_checkpoint(checkpoint)
    if "binding" not in metadata:
        raise ValueError("Unbound legacy checkpoint; use its recorded V1 source revision or train a schema-bound model")
    binding_check = verify_binding(metadata["binding"], dataset_dir/"configurations.json")
    torch.manual_seed(814)
    untrained = Policy(model.obs_size, model.hidden_size, model.feature_indices)
    trained_sources = set(metadata["training_sources"])
    if any(trained_sources & {e["source_episode_id"] for e in configurations[s]} for s in splits if s != "train"):
        raise ValueError("Source leakage in requested evaluation")
    env = RingEnv()
    rows, summaries = [], {}
    rendered = False
    video_case = None
    try:
        for split in splits:
            configs = configurations[split] if limit is None else configurations[split][:limit]
            controllers = ["student", "geometry"] if include_baselines else ["student"]
            if include_baselines:
                controllers += ["zero_action", "untrained_mlp"]
            for controller_name in controllers:
                # Negative controls use one identity configuration per source.
                if controller_name in ("student", "geometry"):
                    cases = configs
                else:
                    seen = set()
                    cases = []
                    for c in configs:
                        if c["source_episode_id"] not in seen:
                            cases.append(c)
                            seen.add(c["source_episode_id"])
                for i, cfg in enumerate(cases):
                    rng = np.random.default_rng(cfg["reset_seed"] + 97)
                    if controller_name == "student":
                        controller = lambda e, o: model.act(noisy_observation(o, rng, noise), normalizer)
                    elif controller_name == "geometry":
                        controller = lambda e, o: teacher(e)
                    elif controller_name == "zero_action":
                        controller = lambda e, o: np.array([0., 0., 0., 1.], dtype=np.float32)
                    else:
                        controller = lambda e, o: untrained.act(noisy_observation(o, rng, noise), normalizer)
                    capture = render_one and controller_name == "student" and not rendered
                    result = run_controller(env, cfg, controller, capture_every=1 if capture else 0)
                    if capture:
                        save_video(output_dir/"student_rollout.mp4", result["images"], fps=round(1/CONTROL_DT))
                        rendered = True
                        video_case = {"configuration_id": cfg["configuration_id"], "split": split,
                                      "source_episode_id": cfg["source_episode_id"],
                                      "success": result["success"], "steps": result["steps"]}
                    row = {"controller": controller_name, "split": split,
                           "configuration_id": cfg["configuration_id"], "source_episode_id": cfg["source_episode_id"],
                           "success": result["success"], "steps": result["steps"], "info": result["info"],
                           "student_teacher_interventions": 0,
                           "observation_noise_std_m": noise if controller_name != "geometry" else 0.}
                    rows.append(row)
                    if (i+1) % 5 == 0 or i+1 == len(cases):
                        successes = sum(r["success"] for r in rows if r["split"] == split and r["controller"] == controller_name)
                        print(f"{split}/{controller_name}: {i+1}/{len(cases)}, successes={successes}", flush=True)
                        write_json(output_dir/"rollouts.json", rows)
                relevant = [r for r in rows if r["split"] == split and r["controller"] == controller_name]
                failures = {}
                for r in relevant:
                    if not r["success"]:
                        reason = r["info"].get("failure_reason", "unknown")
                        failures[reason] = failures.get(reason, 0) + 1
                successes = sum(r["success"] for r in relevant)
                summaries[f"{split}/{controller_name}"] = {"successes": successes, "episodes": len(relevant),
                    "success_rate": successes/len(relevant) if relevant else None,
                    "source_count": len({r["source_episode_id"] for r in relevant}),
                    "failure_counts": failures}
    finally:
        env.close()
        write_json(output_dir/"rollouts.json", rows)
    report = {"checkpoint": str(checkpoint), "training_metadata": metadata, "summaries": summaries,
              "checkpoint_sha256": sha256_file(checkpoint), "binding_verification": binding_check,
              "protocol": manifest.get("protocol"),
              "video_case": video_case,
              "no_teacher_actions_or_rescue_in_student_rollouts": True,
              "success_criteria": {"full_projection_inside_hole": True, "containment_margin_m": .0003,
                  "max_tilt_deg": 8, "table_height_tolerance_m": .0015,
                  "max_linear_speed_m_s": .01, "max_angular_speed_rad_s": .15,
                  "no_finger_contacts": True, "settle_duration_s": env.settle_duration},
              "measurement_noise_note": "Geometry baseline observes exact simulator state; noisy student stress results are separately labeled.",
              "evaluation_scope": "Simulated state-based policy; no real robot transfer or uncalibrated-video 6D recovery claim."}
    write_json(output_dir/"results.json", report)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/dataset_dagger")
    parser.add_argument("--checkpoint", default="outputs/model_dagger/policy.pt")
    parser.add_argument("--output", default="outputs/evaluation_final")
    parser.add_argument("--split", nargs="+", default=["test", "ood_test"])
    parser.add_argument("--noise", type=float, default=0.)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate(args.dataset, args.checkpoint, args.output, args.split, args.noise,
             not args.no_baselines, not args.no_render, args.limit)


if __name__ == "__main__":
    main()
