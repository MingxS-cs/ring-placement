"""Collect teacher corrections at student-visited states of frozen train scenes."""
from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import json
from pathlib import Path
import shutil

import numpy as np

from .dataset import noisy_observation, write_json
from .integrity import environment_signature, sha256_file, validate_split_arrays, verify_binding


def _empty_arrays():
    return {"observations": np.empty((0, 30), dtype=np.float32),
            "actions": np.empty((0, 4), dtype=np.float32),
            "source_episode_ids": np.empty(0, dtype=str),
            "configuration_ids": np.empty(0, dtype=str)}


def validate_training_configurations(manifest: Mapping, configurations: list[dict]) -> list[str]:
    """Reject held-out, duplicated, or modified scenes before collection starts."""
    validate_split_arrays(manifest, "train", _empty_arrays())
    frozen = {cfg["configuration_id"]: cfg for cfg in manifest["configurations"]["train"]}
    if not configurations:
        raise ValueError("At least one frozen train configuration is required")
    seen = set()
    for cfg in configurations:
        configuration_id = cfg.get("configuration_id")
        if cfg.get("split") != "train" or configuration_id not in frozen:
            raise ValueError("Collection configuration must belong to the frozen train manifest")
        if cfg != frozen[configuration_id]:
            raise ValueError("Collection configuration differs from its frozen train configuration")
        if configuration_id in seen:
            raise ValueError("Collection contains a duplicate train configuration")
        seen.add(configuration_id)
    return sorted({cfg["source_episode_id"] for cfg in configurations})


def select_training_configurations(manifest: Mapping, per_source: int = 4,
                                   seed: int = 20261016) -> list[dict]:
    """Select each train source's first identity plus seeded random descendants."""
    if not isinstance(per_source, int) or isinstance(per_source, bool) or per_source < 1:
        raise ValueError("per_source must be a positive integer")
    validate_split_arrays(manifest, "train", _empty_arrays())
    groups = {}
    for cfg in manifest["configurations"]["train"]:
        groups.setdefault(cfg["source_episode_id"], []).append(cfg)
    rng = np.random.default_rng(seed)
    selected = []
    for source in sorted(groups):
        group = groups[source]
        if len(group) < per_source:
            raise ValueError(f"per_source exceeds frozen train configurations for {source!r}")
        identities = [cfg for cfg in group if cfg.get("augmentation", {}).get("identity") is True]
        if not identities:
            raise ValueError(f"Train source {source!r} has no frozen identity configuration")
        identity = identities[0]
        others = [cfg for cfg in group if cfg["configuration_id"] != identity["configuration_id"]]
        indices = rng.choice(len(others), size=per_source - 1, replace=False)
        selected.extend([identity, *(others[int(index)] for index in indices)])
    validate_training_configurations(manifest, selected)
    return copy.deepcopy(selected)


def collect_corrections(env, manifest: Mapping, configurations: list[dict], student,
                        teacher_action, *, beta: float = .25, seed: int = 20261016,
                        observation_noise: float = .0002, max_steps: int = 450):
    """Label every visited train state, including failed mixed-policy rollouts.

    Student measurement noise never alters the environment or the clean saved
    observations. The teacher labels the current physical state before either
    controller advances it; ``beta`` is the teacher execution probability.
    """
    validate_training_configurations(manifest, configurations)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be a finite probability between zero and one")
    if not np.isfinite(observation_noise) or observation_noise < 0:
        raise ValueError("observation_noise must be finite and nonnegative")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    mixture_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    observations, labels, source_ids, configuration_ids, logs = [], [], [], [], []
    for index, cfg in enumerate(configurations):
        obs = env.reset(cfg["ring_xy_m"], cfg["bearing_xy_m"], seed=cfg["reset_seed"])
        interventions, info = 0, {}
        for step in range(max_steps):
            clean_obs = np.array(obs, dtype=np.float32, copy=True)
            label = np.asarray(teacher_action(env), dtype=np.float32)
            if clean_obs.shape != (30,) or not np.isfinite(clean_obs).all():
                raise ValueError("Environment returned invalid observation")
            if label.shape != (4,) or not np.isfinite(label).all() or np.max(np.abs(label)) > 1:
                raise ValueError("Teacher returned invalid normalized action")
            observations.append(clean_obs)
            labels.append(label.copy())
            source_ids.append(cfg["source_episode_id"])
            configuration_ids.append(cfg["configuration_id"])
            intervene = mixture_rng.random() < beta
            if intervene:
                action = label
                interventions += 1
            else:
                action = np.asarray(student(noisy_observation(clean_obs, noise_rng,
                                                              observation_noise)), dtype=np.float32)
                if action.shape != (4,) or not np.isfinite(action).all():
                    raise ValueError("Student returned invalid action")
            obs, done, info = env.step(np.clip(action, -1., 1.))
            if done:
                break
        logs.append({"configuration_id": cfg["configuration_id"],
                     "source_episode_id": cfg["source_episode_id"], "split": "train",
                     "success": bool(env.success()), "steps": step + 1, "info": info,
                     "teacher_interventions": interventions,
                     "student_steps": step + 1 - interventions})
        print(f"train/DAgger {index + 1}/{len(configurations)}, "
              f"steps={step + 1}, success={logs[-1]['success']}, "
              f"teacher_interventions={interventions}", flush=True)
    arrays = {"observations": np.stack(observations), "actions": np.stack(labels),
              "source_episode_ids": np.asarray(source_ids, dtype=str),
              "configuration_ids": np.asarray(configuration_ids, dtype=str)}
    validate_split_arrays(manifest, "train", arrays)
    return arrays, logs


def collect(dataset_dir, checkpoint, output_dir, per_source: int = 4, beta: float = .25,
            seed: int = 20261016, observation_noise: float = .0002, max_steps: int = 450):
    """Append train corrections without regenerating or changing held-out data."""
    from .policy import load_checkpoint

    dataset_dir, checkpoint, output_dir = Path(dataset_dir), Path(checkpoint), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("DAgger output already exists; choose a new dataset directory")
    manifest_path = dataset_dir / "configurations.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("environment") != environment_signature():
        raise ValueError("Dataset environment does not match the current simulation/teacher")
    model, normalizer, metadata = load_checkpoint(checkpoint)
    binding_check = verify_binding(metadata.get("binding"), manifest_path)
    checkpoint_hash = sha256_file(checkpoint)
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as data:
        validate_split_arrays(manifest, "train", data)
        original = {key: data[key].copy() for key in _empty_arrays()}
    # Structural validation does not execute validation states or use their labels
    # to select collection settings. The original NPZ is copied without rewriting.
    with np.load(dataset_dir / "val.npz", allow_pickle=False) as data:
        validate_split_arrays(manifest, "val", data)
    configurations = select_training_configurations(manifest, per_source, seed)
    sources = validate_training_configurations(manifest, configurations)

    from .sim import RingEnv
    from .teacher import teacher

    env = RingEnv(max_steps=max_steps)
    try:
        corrections, logs = collect_corrections(env, manifest, configurations,
                                               lambda obs: model.act(obs, normalizer), teacher,
                                               beta=beta, seed=seed,
                                               observation_noise=observation_noise,
                                               max_steps=max_steps)
    finally:
        env.close()
    combined = {key: np.concatenate([original[key], corrections[key]]) for key in original}
    validate_split_arrays(manifest, "train", corrections)
    validate_split_arrays(manifest, "train", combined)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output_dir / "configurations.json")
    shutil.copyfile(dataset_dir / "val.npz", output_dir / "val.npz")
    np.savez_compressed(output_dir / "train.npz", **combined)
    report = {"algorithm": "train_only_DAgger", "source_dataset": str(dataset_dir.resolve()),
              "baseline_checkpoint": str(checkpoint.resolve()),
              "baseline_checkpoint_sha256": checkpoint_hash,
              "binding_verification": binding_check,
              "manifest_sha256": sha256_file(output_dir / "configurations.json"),
              "validation_npz_sha256": sha256_file(output_dir / "val.npz"),
              "original_training_npz_sha256": sha256_file(dataset_dir / "train.npz"),
              "per_source": per_source, "beta": beta, "seed": seed,
              "observation_noise_std_m": observation_noise,
              "source_episode_ids": sources,
              "original_train_rows": len(original["observations"]),
              "correction_row_count": len(corrections["observations"]),
              "combined_train_rows": len(combined["observations"]),
              "teacher_interventions": sum(row["teacher_interventions"] for row in logs),
              "student_steps": sum(row["student_steps"] for row in logs),
              "all_label_states_train_only": True,
              "teacher_labels_at_every_visited_state": True,
              "manifest_and_validation_bytes_preserved": True,
              "held_out_rollouts_used_for_collection": False,
              "rollouts": logs}
    write_json(output_dir / "dagger_collection.json", report)
    print(f"DAgger: {len(logs)} train scenes, {report['correction_row_count']} corrections, "
          f"{report['teacher_interventions']} teacher interventions", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/dataset_v2")
    parser.add_argument("--checkpoint", default="outputs/model_v2/policy.pt")
    parser.add_argument("--output", default="outputs/dataset_dagger")
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--noise", type=float, default=.0002)
    parser.add_argument("--max-steps", type=int, default=450)
    args = parser.parse_args()
    collect(args.dataset, args.checkpoint, args.output, args.per_source, args.beta,
            args.seed, args.noise, args.max_steps)


if __name__ == "__main__":
    main()
