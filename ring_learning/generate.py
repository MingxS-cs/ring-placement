"""Generate synthetic robot actions from video-derived scene configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .dataset import make_configurations, split_episodes, validate_lineage, write_json
from .integrity import environment_signature


def run_controller(env, configuration, controller, max_steps=500, capture_every=0):
    obs = env.reset(configuration["ring_xy_m"], configuration["bearing_xy_m"],
                    seed=configuration["reset_seed"])
    observations, actions, images = [], [], []
    info = {}
    for step in range(max_steps):
        action = np.asarray(controller(env, obs), dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Controller returned invalid action")
        action = np.clip(action, -1., 1.)
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(action.copy())
        obs, done, info = env.step(action)
        if capture_every and step % capture_every == 0:
            images.append(env.render())
        if done:
            break
    return {"observations": np.stack(observations), "actions": np.stack(actions),
            "success": bool(env.success()), "steps": step + 1, "info": info,
            "images": images}


def generate(data_path, output_dir, augment=16, seed=20261014, force=False,
             rotation_deg=30., translation=.010, configuration_seed=None, fresh_tests=False):
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
    configuration_seed = seed if configuration_seed is None else configuration_seed
    configurations = make_configurations(splits, counts, seed=configuration_seed,
                                        angle_limit=np.deg2rad(rotation_deg), translation_limit=translation)
    if fresh_tests:
        # Frozen before teacher generation and training. These are new simulated
        # scenes descended only from held-out real sources, after V1 diagnostics.
        rng = np.random.default_rng(982451653)
        for split in ("test", "ood_test"):
            for cfg in configurations[split]:
                old_bearing = np.asarray(cfg["bearing_xy_m"])
                delta = np.asarray(cfg["ring_xy_m"]) - old_bearing
                new_bearing = rng.uniform([.40, -.10], [.52, .10])
                cfg["bearing_xy_m"] = new_bearing.tolist()
                cfg["ring_xy_m"] = (new_bearing + delta).tolist()
                cfg["augmentation"]["fresh_test_target_bounds_m"] = [[.40, -.10], [.52, .10]]
                cfg["augmentation"]["bearing_relocation_m"] = (new_bearing-old_bearing).tolist()
                cfg["augmentation"]["identity"] = False
                cfg["fresh_test_scene"] = True
    validate_lineage(configurations)
    manifest = {"seed": seed, "configuration_seed": configuration_seed,
                "environment": environment_signature(),
                "source_counts": {s: len(e) for s, e in splits.items()},
                "configurations": configurations, "augmentation_bounds": {
                    "rotation_deg": rotation_deg, "translation_m": translation, "reset_jitter_m": .001,
                    "object_geometry_scale": 1.0},
                "robot_label_source": "MuJoCo teacher using video-derived configurations",
                "metric_mapping": "uncalibrated known-diameter approximation",
                "evaluation_configurations_not_used_for_training": True,
                "protocol": "fresh_simulated_scenes_frozen_after_V1_diagnostics" if fresh_tests else "original_source_group_holdout",
                "fresh_test_seed": 982451653 if fresh_tests else None,
                "fresh_test_note": "New seeded simulated scenes; correlated descendants of original held-out sources, not independent new human recordings." if fresh_tests else None}
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
    parser.add_argument("--rotation", type=float, default=30.)
    parser.add_argument("--translation", type=float, default=.010)
    parser.add_argument("--configuration-seed", type=int)
    parser.add_argument("--fresh-tests", action="store_true")
    args = parser.parse_args()
    generate(args.data, args.output, args.augment, args.seed, args.force,
             args.rotation, args.translation, args.configuration_seed, args.fresh_tests)


if __name__ == "__main__":
    main()
