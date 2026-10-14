"""Validate sample lineage and bind checkpoints to data and simulation sources.

Signatures read source files rather than importing simulation modules, so these
checks do not load MuJoCo or construct an environment.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
from pathlib import Path


SCHEMA_VERSION = 1
OBS_SIZE = 30
ACTION_SIZE = 4
_SOURCE_ROOT = Path(__file__).resolve().parent
_ENVIRONMENT_KEYS = {
    "schema_version", "sim_sha256", "teacher_sha256", "obs_size", "action_size",
    "action_scale", "control_dt",
}


def sha256_file(path: str | Path) -> str:
    """Hash the exact saved bytes, including manifest formatting."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_arrays(manifest: Mapping, split: str, arrays) -> list[str]:
    """Reject cross-split lineage or rows outside their declared source pair.

    Accepts a mapping or an ``np.load(..., allow_pickle=False)`` result. Returns
    the sorted distinct source episode IDs represented by the validated rows.
    Manifest checks cover every split, including validation/test isolation.
    """
    import numpy as np

    configurations = manifest.get("configurations") if isinstance(manifest, Mapping) else None
    if not isinstance(configurations, Mapping) or split not in configurations:
        raise ValueError(f"Manifest has no configurations for split {split!r}")

    configuration_owners = {}
    source_owners = {}
    pairs = set()
    for owner, group in configurations.items():
        if not isinstance(group, list):
            raise ValueError(f"Manifest configurations for split {owner!r} must be a list")
        for configuration in group:
            if not isinstance(configuration, Mapping):
                raise ValueError("Manifest configuration must be a mapping")
            configuration_id = configuration.get("configuration_id")
            source_id = configuration.get("source_episode_id")
            if not isinstance(configuration_id, str) or not configuration_id:
                raise ValueError("Manifest configuration ID must be a nonempty string")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Manifest source episode ID must be a nonempty string")
            if configuration.get("split") != owner:
                raise ValueError(f"Configuration {configuration_id!r} has an inconsistent split")
            if configuration_id in configuration_owners:
                raise ValueError(f"Duplicate configuration ID: {configuration_id!r}")
            configuration_owners[configuration_id] = owner
            if source_id in source_owners and source_owners[source_id] != owner:
                raise ValueError(f"Source leakage: {source_id!r} belongs to multiple splits")
            source_owners[source_id] = owner
            if owner == split:
                pairs.add((configuration_id, source_id))

    required = ("observations", "actions", "source_episode_ids", "configuration_ids")
    try:
        observations, actions, source_ids, configuration_ids = (
            np.asarray(arrays[key]) for key in required)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Split {split!r} is missing a required sample array") from error

    if observations.ndim != 2 or observations.shape[1] != OBS_SIZE:
        raise ValueError(f"observations shape must be (N, {OBS_SIZE})")
    row_count = len(observations)
    if actions.shape != (row_count, ACTION_SIZE):
        raise ValueError(f"actions shape must be (N, {ACTION_SIZE}) with matching rows")
    if source_ids.shape != (row_count,) or configuration_ids.shape != (row_count,):
        raise ValueError("source_episode_ids and configuration_ids shape must be (N,)")
    for name, values in (("observations", observations), ("actions", actions)):
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite numeric values")
    if source_ids.dtype.kind != "U" or configuration_ids.dtype.kind != "U":
        raise ValueError("Sample source/configuration IDs must be string arrays")
    for configuration_id, source_id in set(zip(configuration_ids.tolist(), source_ids.tolist())):
        if (configuration_id, source_id) not in pairs:
            raise ValueError(
                f"Sample pair ({configuration_id!r}, {source_id!r}) does not belong to split {split!r}")
    return sorted(set(source_ids.tolist()))


def environment_signature() -> dict:
    """Describe the current observation/action contract and normalized sources."""
    sources = {
        name: (_SOURCE_ROOT / f"{name}.py").read_text(encoding="utf-8-sig")
        .replace("\r\n", "\n").replace("\r", "\n")
        for name in ("sim", "teacher")
    }
    constants = {}
    wanted = {"OBS_SIZE", "ACTION_SIZE", "ACTION_SCALE", "CONTROL_DT"}
    for statement in ast.parse(sources["sim"]).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = ast.literal_eval(statement.value)
    if constants.keys() != wanted:
        raise ValueError("Simulation source is missing observation/action schema constants")
    return {
        "schema_version": SCHEMA_VERSION,
        "sim_sha256": hashlib.sha256(sources["sim"].encode("utf-8")).hexdigest(),
        "teacher_sha256": hashlib.sha256(sources["teacher"].encode("utf-8")).hexdigest(),
        "obs_size": constants["OBS_SIZE"],
        "action_size": constants["ACTION_SIZE"],
        "action_scale": constants["ACTION_SCALE"],
        "control_dt": constants["CONTROL_DT"],
    }


def binding_for_manifest(manifest_path: str | Path) -> dict:
    """Create checkpoint metadata without touching policy weights or the data."""
    return {"schema_version": SCHEMA_VERSION,
            "manifest_sha256": sha256_file(manifest_path),
            "environment": environment_signature()}


def verify_binding(saved: Mapping | None, manifest_path: str | Path, *,
                   allow_environment_change: bool = False) -> dict:
    """Verify saved provenance and report a deliberate environment override.

    Data and schema mismatches always fail. An explicit environment override is
    reported in the return value and never changes the saved checkpoint binding.
    """
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint binding is missing; legacy checkpoints require explicit handling")
    if not {"schema_version", "manifest_sha256", "environment"}.issubset(saved):
        raise ValueError("Checkpoint binding is incomplete")
    if saved["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Checkpoint binding schema is incompatible")
    saved_environment = saved["environment"]
    if not isinstance(saved_environment, Mapping) or not _ENVIRONMENT_KEYS.issubset(saved_environment):
        raise ValueError("Checkpoint environment binding is incomplete")
    current = binding_for_manifest(manifest_path)
    if saved["manifest_sha256"] != current["manifest_sha256"]:
        raise ValueError("Checkpoint manifest hash differs from the current data manifest")
    for key in ("schema_version", "obs_size", "action_size"):
        if saved_environment[key] != current["environment"][key]:
            raise ValueError(f"Checkpoint environment schema is incompatible: {key}")
    changed = dict(saved_environment) != current["environment"]
    if changed and not allow_environment_change:
        raise ValueError("Checkpoint environment differs from the current simulation/teacher; explicit override required")
    return {"current_binding": current,
            "saved_environment": dict(saved_environment),
            "environment_changed": changed,
            "environment_change_allowed": bool(changed and allow_environment_change)}
