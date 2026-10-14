"""Generate synthetic robot actions from video-derived scene configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .dataset import make_configurations, split_episodes, validate_lineage, write_json


def run_controller(env, configuration, controller, max_steps=500, capture_every=0):
    obs = env.reset(configuration["ring_xy_m"], configuration["bearing_xy_m"],
                    seed=configuration["reset_seed"])
    observations, actions, images = [], [], []
    info = {}
    for step in range(max_steps):
        action = np.asarray(controller(env, obs), dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Controller returned invalid action")
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(action.copy())
        obs, done, info = env.step(np.clip(action, -1., 1.))
        if capture_every and step % capture_every == 0:
            images.append(env.render())
        if done:
            break
    return {"observations": np.stack(observations), "actions": np.stack(actions),
            "success": bool(env.success()), "steps": step + 1, "info": info,
            "images": images}


def generate(data_path, output_dir, augment=16, seed=20261014, force=False):
    from .sim import RingEnv
    from .teacher import teacher

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "configurations.json"
    if manifest_path.exists() and not force:
        raise FileExistsError("Configurations already frozen; use a new output directory or --force")
    real = json.loads(Path(data_path).read_text(encoding="utf-8"))
    episodes = real["episodes"]
    splits = split_episodes(episodes, seed)
    counts = {"train": augment, "val": max(3, augment // 4), "test": 6, "ood_test": 6}
    configurations = make_configurations(splits, counts, seed=seed)
    validate_lineage(configurations)
    manifest = {"seed": seed, "source_counts": {s: len(e) for s, e in splits.items()},
                "configurations": configurations, "augmentation_bounds": {
                    "rotation_deg": 30, "translation_m": .010, "reset_jitter_m": .001,
                    "object_geometry_scale": 1.0},
                "robot_label_source": "MuJoCo teacher using video-derived configurations",
                "metric_mapping": "uncalibrated known-diameter approximation",
                "evaluation_configurations_not_used_for_training": True}
    write_json(manifest_path, manifest)
    env = RingEnv()
    start = time.monotonic()
    logs = []
    try:
        for split in ("train", "val"):
            all_obs, all_actions, ids, config_ids = [], [], [], []
            for i, cfg in enumerate(configurations[split]):
                rollout = run_controller(env, cfg, lambda e, o: teacher(e))
                log = {"configuration_id": cfg["configuration_id"],
                       "source_episode_id": cfg["source_episode_id"], "split": split,
                       "success": rollout["success"], "steps": rollout["steps"],
                       "info": rollout["info"]}
                logs.append(log)
                if rollout["success"]:
                    all_obs.append(rollout["observations"])
                    all_actions.append(rollout["actions"])
                    ids.extend([cfg["source_episode_id"]] * rollout["steps"])
                    config_ids.extend([cfg["configuration_id"]] * rollout["steps"])
                if (i + 1) % 5 == 0 or i + 1 == len(configurations[split]):
                    successes = sum(r["success"] for r in logs if r["split"] == split)
                    print(f"{split} {i+1}/{len(configurations[split])}, successful={successes}, elapsed={time.monotonic()-start:.1f}s", flush=True)
                    write_json(output_dir / "teacher_generation.json", logs)
            if not all_obs:
                raise RuntimeError(f"No successful {split} teacher rollouts; inspect teacher_generation.json")
            np.savez_compressed(output_dir / f"{split}.npz", observations=np.concatenate(all_obs),
                                actions=np.concatenate(all_actions), source_episode_ids=np.asarray(ids),
                                configuration_ids=np.asarray(config_ids))
    finally:
        env.close()
        write_json(output_dir / "teacher_generation.json", logs)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/real_episodes.json")
    parser.add_argument("--output", default="outputs/dataset")
    parser.add_argument("--augment", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(args.data, args.output, args.augment, args.seed, args.force)


if __name__ == "__main__":
    main()
