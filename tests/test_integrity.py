import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def manifest():
    return {
        "configurations": {
            split: [{"configuration_id": f"{source}_{split}_000",
                     "source_episode_id": source, "split": split}]
            for split, source in (("train", "E01"), ("val", "E02"),
                                  ("test", "E03"), ("ood_test", "E04"))
        }
    }


def arrays(split="train", source="E01"):
    return {
        "observations": np.zeros((2, 30), dtype=np.float32),
        "actions": np.zeros((2, 4), dtype=np.float32),
        "source_episode_ids": np.array([source, source]),
        "configuration_ids": np.array([f"{source}_{split}_000"] * 2),
    }


def test_normalized_action_labels_reject_out_of_range_values():
    from ring_learning.integrity import validate_split_arrays
    rows = arrays()
    rows["actions"][0, 0] = 1.25
    with pytest.raises(ValueError, match="normalized"):
        validate_split_arrays(manifest(), "train", rows)


def test_split_rows_must_match_manifest_configuration_source_pairs():
    from ring_learning.integrity import validate_split_arrays

    data = manifest()
    data["configurations"]["train"].append({
        "configuration_id": "E05_train_000", "source_episode_id": "E05", "split": "train"})
    rows = arrays()
    rows["source_episode_ids"][0] = "E05"
    with pytest.raises(ValueError, match="pair"):
        validate_split_arrays(data, "train", rows)


@pytest.mark.parametrize("first,second", [("train", "val"), ("train", "test"),
                                           ("train", "ood_test"), ("val", "test"),
                                           ("val", "ood_test"), ("test", "ood_test")])
def test_manifest_rejects_source_leakage_between_every_split_pair(first, second):
    from ring_learning.integrity import validate_split_arrays

    data = manifest()
    data["configurations"][second][0]["source_episode_id"] = data["configurations"][first][0]["source_episode_id"]
    with pytest.raises(ValueError, match="Source leakage"):
        validate_split_arrays(data, "train", arrays())


def test_manifest_rejects_duplicate_configuration_ids():
    from ring_learning.integrity import validate_split_arrays

    data = manifest()
    data["configurations"]["test"][0]["configuration_id"] = "E02_val_000"
    with pytest.raises(ValueError, match="configuration ID"):
        validate_split_arrays(data, "train", arrays())


def test_manifest_rejects_configuration_placed_under_wrong_split():
    from ring_learning.integrity import validate_split_arrays

    data = manifest()
    data["configurations"]["val"][0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        validate_split_arrays(data, "train", arrays())


@pytest.mark.parametrize("key,bad", [
    ("observations", np.zeros((2, 29))),
    ("observations", np.zeros((2, 30, 1))),
    ("actions", np.zeros((2, 3))),
    ("actions", np.zeros((1, 4))),
    ("source_episode_ids", np.array(["E01"])),
    ("configuration_ids", np.array([["E01_train_000", "E01_train_000"]])),
])
def test_split_rejects_incompatible_row_shapes(key, bad):
    from ring_learning.integrity import validate_split_arrays

    rows = arrays()
    rows[key] = bad
    with pytest.raises(ValueError, match="shape"):
        validate_split_arrays(manifest(), "train", rows)


@pytest.mark.parametrize("key,value", [("observations", np.nan), ("observations", np.inf),
                                      ("actions", -np.inf)])
def test_split_rejects_nonfinite_values(key, value):
    from ring_learning.integrity import validate_split_arrays

    rows = arrays()
    rows[key][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        validate_split_arrays(manifest(), "train", rows)


def test_split_rejects_configuration_from_another_split():
    from ring_learning.integrity import validate_split_arrays

    rows = arrays("val", "E02")
    with pytest.raises(ValueError, match="pair"):
        validate_split_arrays(manifest(), "train", rows)


def test_valid_npz_rows_return_only_distinct_sources_in_that_split(tmp_path):
    from ring_learning.integrity import validate_split_arrays

    path = tmp_path / "train.npz"
    np.savez(path, **arrays())
    with np.load(path, allow_pickle=False) as rows:
        assert validate_split_arrays(manifest(), "train", rows) == ["E01"]


def test_sha256_file_hashes_raw_bytes(tmp_path):
    from ring_learning.integrity import sha256_file

    path = tmp_path / "manifest.json"
    content = b'{"a": 1}\r\n'
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
    path.write_bytes(b'{"a": 2}\r\n')
    assert sha256_file(path) != hashlib.sha256(content).hexdigest()


def test_environment_signature_hashes_normalized_sources_and_schema():
    from ring_learning.integrity import environment_signature

    signature = environment_signature()
    root = Path(__file__).resolve().parents[1] / "ring_learning"
    for name in ("sim", "teacher"):
        source = (root / f"{name}.py").read_text(encoding="utf-8-sig")
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        assert signature[f"{name}_sha256"] == hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    sim_ast = ast.parse((root / "sim.py").read_text(encoding="utf-8-sig"))
    schema_assignments = [node for node in sim_ast.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "OBS_SCHEMA_VERSION"
                                  for target in node.targets)]
    expected_schema = ast.literal_eval(schema_assignments[0].value) if schema_assignments else 1
    assert signature["schema_version"] == expected_schema
    assert signature["obs_size"] == 30
    assert signature["action_size"] == 4
    assert signature["action_scale"] == 0.003
    assert signature["control_dt"] == 0.05


@pytest.mark.parametrize("schema", [None, 2])
def test_signature_derives_observation_schema_version_without_importing_source(tmp_path, monkeypatch, schema):
    from ring_learning import integrity

    sim_source = "OBS_SIZE = 30\nACTION_SIZE = 4\nACTION_SCALE = .003\nCONTROL_DT = .05\n"
    if schema is not None:
        sim_source += f"OBS_SCHEMA_VERSION = {schema}\n"
    sim_source += "raise RuntimeError('source must not be imported')\n"
    (tmp_path / "sim.py").write_bytes(b"\xef\xbb\xbf" + sim_source.replace("\n", "\r\n").encode("utf-8"))
    (tmp_path / "teacher.py").write_text("raise RuntimeError('source must not be imported')\n", encoding="utf-8")
    monkeypatch.setattr(integrity, "_SOURCE_ROOT", tmp_path)
    signature = integrity.environment_signature()
    assert signature["schema_version"] == (1 if schema is None else schema)
    assert signature["sim_sha256"] == hashlib.sha256(sim_source.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    binding = integrity.binding_for_manifest(manifest_path)
    assert binding["schema_version"] == signature["schema_version"]
    assert integrity.verify_binding(binding, manifest_path)["environment_changed"] is False


def test_signature_does_not_import_mujoco_or_construct_environment():
    root = Path(__file__).resolve().parents[1]
    program = "import sys; from ring_learning.integrity import environment_signature; environment_signature(); assert 'mujoco' not in sys.modules; assert 'ring_learning.sim' not in sys.modules; assert 'ring_learning.teacher' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", program], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_binding_round_trip_is_verified(tmp_path):
    from ring_learning.integrity import binding_for_manifest, sha256_file, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    binding = binding_for_manifest(path)
    assert binding["schema_version"] == binding["environment"]["schema_version"]
    assert binding["manifest_sha256"] == sha256_file(path)
    verified = verify_binding(binding, path)
    assert verified["environment_changed"] is False
    assert verified["environment_change_allowed"] is False
    assert verified["current_binding"] == binding


@pytest.mark.parametrize("allow_environment_change", [False, True])
def test_binding_rejects_changed_manifest_even_if_environment_override_is_allowed(tmp_path, allow_environment_change):
    from ring_learning.integrity import binding_for_manifest, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    binding = binding_for_manifest(path)
    path.write_text(json.dumps({**manifest(), "seed": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        verify_binding(binding, path, allow_environment_change=allow_environment_change)


def test_binding_rejects_changed_environment_without_explicit_override(tmp_path):
    from ring_learning.integrity import binding_for_manifest, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    binding = binding_for_manifest(path)
    binding["environment"]["sim_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="environment"):
        verify_binding(binding, path)


def test_binding_records_explicit_environment_override_without_mutating_saved_binding(tmp_path):
    from ring_learning.integrity import binding_for_manifest, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    binding = binding_for_manifest(path)
    binding["environment"]["teacher_sha256"] = "0" * 64
    original = copy.deepcopy(binding)
    verified = verify_binding(binding, path, allow_environment_change=True)
    assert verified["environment_changed"] is True
    assert verified["environment_change_allowed"] is True
    assert verified["saved_environment"] == original["environment"]
    assert verified["current_binding"]["environment"] != original["environment"]
    assert binding == original


@pytest.mark.parametrize("key,new_value", [("schema_version", 999), ("obs_size", 29), ("action_size", 3)])
def test_binding_cannot_override_schema_incompatibility(tmp_path, key, new_value):
    from ring_learning.integrity import binding_for_manifest, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    binding = binding_for_manifest(path)
    binding["environment"][key] = new_value
    with pytest.raises(ValueError, match="schema"):
        verify_binding(binding, path, allow_environment_change=True)


def test_binding_rejects_legacy_or_incomplete_metadata(tmp_path):
    from ring_learning.integrity import binding_for_manifest, verify_binding

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        verify_binding(None, path)
    binding = binding_for_manifest(path)
    del binding["environment"]["teacher_sha256"]
    with pytest.raises(ValueError, match="binding"):
        verify_binding(binding, path, allow_environment_change=True)
