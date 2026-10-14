import copy
import importlib
import json

import numpy as np
import pytest


def dagger_module():
    assert importlib.util.find_spec("ring_learning.dagger") is not None, "DAgger collector is missing"
    return importlib.import_module("ring_learning.dagger")


def manifest():
    groups = {}
    for split, sources in (("train", ("E01", "E05")), ("val", ("E02",)),
                           ("test", ("E03",)), ("ood_test", ("E04",))):
        groups[split] = [
            {"configuration_id": f"{source}_{split}_{i:03d}",
             "source_episode_id": source, "split": split,
             "ring_xy_m": [.39 + i * .001, -.07], "bearing_xy_m": [.48, .02],
             "reset_seed": 100 + i, "augmentation": {"identity": i == 0}}
            for source in sources for i in range(5)
        ]
    return {"configurations": groups}


def test_selection_is_seeded_with_one_identity_and_only_training_sources():
    dagger = dagger_module()
    frozen = manifest()
    before = copy.deepcopy(frozen)
    selected = dagger.select_training_configurations(frozen, per_source=3, seed=11)
    assert selected == dagger.select_training_configurations(frozen, per_source=3, seed=11)
    assert len(selected) == 6
    assert {cfg["source_episode_id"] for cfg in selected} == {"E01", "E05"}
    assert all(cfg["split"] == "train" for cfg in selected)
    for source in ("E01", "E05"):
        cases = [cfg for cfg in selected if cfg["source_episode_id"] == source]
        assert len(cases) == 3
        assert cases[0]["augmentation"]["identity"] is True
        assert len({cfg["configuration_id"] for cfg in cases}) == 3
    selected[0]["ring_xy_m"][0] = 9
    assert frozen == before


@pytest.mark.parametrize("split", ["val", "test", "ood_test"])
def test_selection_validation_rejects_held_out_configuration(split):
    dagger = dagger_module()
    frozen = manifest()
    with pytest.raises(ValueError, match="train"):
        dagger.validate_training_configurations(frozen, [frozen["configurations"][split][0]])


@pytest.mark.parametrize("change", ["wrong_source", "changed_geometry", "pretend_train"])
def test_selection_validation_rejects_forged_or_modified_configuration(change):
    dagger = dagger_module()
    frozen = manifest()
    cfg = copy.deepcopy(frozen["configurations"]["train"][0])
    if change == "wrong_source":
        cfg["source_episode_id"] = "E02"
    elif change == "changed_geometry":
        cfg["ring_xy_m"][0] += .01
    else:
        cfg = copy.deepcopy(frozen["configurations"]["test"][0])
        cfg["split"] = "train"
    with pytest.raises(ValueError, match="train|frozen"):
        dagger.validate_training_configurations(frozen, [cfg])


def test_selection_rejects_manifest_source_leakage():
    dagger = dagger_module()
    frozen = manifest()
    frozen["configurations"]["test"][0]["source_episode_id"] = "E01"
    with pytest.raises(ValueError, match="Source leakage"):
        dagger.select_training_configurations(frozen, per_source=1)


@pytest.mark.parametrize("per_source", [0, -1, 6])
def test_selection_rejects_invalid_per_source_count(per_source):
    dagger = dagger_module()
    with pytest.raises(ValueError, match="per_source"):
        dagger.select_training_configurations(manifest(), per_source=per_source)


class CounterEnv:
    """Small deterministic state transition for testing on-policy label timing."""
    def __init__(self):
        self.reset_positions = []
        self.executed_actions = []
        self.steps = 0

    def observe(self):
        obs = np.zeros(30, dtype=np.float32)
        obs[:3] = [self.position, .02, .15]
        obs[3:6] = [.45, .02, .1035]
        obs[6:9] = [.48, .02, .1025]
        obs[9:12] = obs[3:6] - obs[:3]
        obs[12:15] = obs[6:9] - obs[3:6]
        obs[15:18] = [.001, -.002, .003]
        obs[22] = 1
        return obs

    def reset(self, ring_xy, bearing_xy, seed):
        self.reset_positions.append((ring_xy, bearing_xy, seed))
        self.position = ring_xy[0]
        self.steps = 0
        return self.observe()

    def step(self, action):
        self.executed_actions.append(np.array(action, copy=True))
        self.position += float(action[0])
        self.steps += 1
        return self.observe(), self.steps == 3, {"success": False, "steps": self.steps}

    def success(self):
        return False


def correction_label(env):
    return np.array([env.position / 10, .1, -.2, 1], dtype=np.float32)


def test_student_rollout_labels_every_current_state_and_uses_noise_only_for_student():
    dagger = dagger_module()
    assert hasattr(dagger, "collect_corrections"), "On-policy correction rollout is missing"
    frozen = manifest()
    selected = dagger.select_training_configurations(frozen, per_source=1)
    env, student_inputs = CounterEnv(), []

    def student(obs):
        student_inputs.append(obs.copy())
        return np.array([.3, -.2, .1, -1], dtype=np.float32)

    rows, logs = dagger.collect_corrections(env, frozen, selected, student, correction_label,
                                          beta=0., seed=31, observation_noise=.0002)
    assert rows["observations"].shape == (6, 30)
    assert rows["actions"].shape == (6, 4)
    np.testing.assert_allclose(rows["observations"][:, 0], [.39, .69, .99] * 2, atol=1e-6)
    np.testing.assert_allclose(rows["actions"][:, 0], rows["observations"][:, 0] / 10)
    assert len(student_inputs) == 6
    assert not np.array_equal(student_inputs, rows["observations"])
    assert all(np.max(np.abs(noisy[:9] - clean[:9])) <= .0006001
               for noisy, clean in zip(student_inputs, rows["observations"]))
    assert len(env.reset_positions) == 2
    assert all(log["teacher_interventions"] == 0 and log["student_steps"] == 3 for log in logs)
    assert all(log["success"] is False and log["steps"] == 3 for log in logs)
    assert rows["source_episode_ids"].dtype.kind == "U"
    assert rows["configuration_ids"].dtype.kind == "U"
    assert set(rows["source_episode_ids"]) == {"E01", "E05"}


def test_teacher_mixture_executes_label_when_beta_is_one():
    dagger = dagger_module()
    assert hasattr(dagger, "collect_corrections"), "On-policy correction rollout is missing"
    frozen = manifest()
    selected = dagger.select_training_configurations(frozen, per_source=1)
    env = CounterEnv()

    def unexpected_student(obs):
        raise AssertionError("beta=1 should always execute teacher action")

    rows, logs = dagger.collect_corrections(env, frozen, selected, unexpected_student,
                                          correction_label, beta=1., seed=4)
    np.testing.assert_array_equal(env.executed_actions, rows["actions"])
    assert all(log["teacher_interventions"] == 3 and log["student_steps"] == 0 for log in logs)


@pytest.mark.parametrize("split", ["val", "test", "ood_test"])
def test_rollout_rejects_held_out_configurations_before_environment_reset(split):
    dagger = dagger_module()
    assert hasattr(dagger, "collect_corrections"), "On-policy correction rollout is missing"
    frozen, env = manifest(), CounterEnv()
    with pytest.raises(ValueError, match="train"):
        dagger.collect_corrections(env, frozen, frozen["configurations"][split][:1],
                                   lambda obs: np.zeros(4), correction_label)
    assert env.reset_positions == []


@pytest.mark.parametrize("beta", [-.1, 1.1, np.nan, np.inf])
def test_rollout_rejects_invalid_teacher_probability_before_reset(beta):
    dagger = dagger_module()
    assert hasattr(dagger, "collect_corrections"), "On-policy correction rollout is missing"
    frozen, env = manifest(), CounterEnv()
    selected = dagger.select_training_configurations(frozen, per_source=1)
    with pytest.raises(ValueError, match="beta"):
        dagger.collect_corrections(env, frozen, selected, lambda obs: np.zeros(4),
                                   correction_label, beta=beta)
    assert env.reset_positions == []


def small_bound_dataset(tmp_path):
    from ring_learning.dataset import Normalizer
    from ring_learning.integrity import binding_for_manifest, environment_signature
    from ring_learning.policy import Policy, save_checkpoint

    dataset = tmp_path / "source_dataset"
    dataset.mkdir()
    frozen = {**manifest(), "environment": environment_signature()}
    manifest_path = dataset / "configurations.json"
    manifest_path.write_bytes(json.dumps(frozen, indent=3).replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    for split, sources in (("train", ["E01", "E05"]), ("val", ["E02"])):
        np.savez_compressed(dataset / f"{split}.npz",
                            observations=np.zeros((len(sources), 30), dtype=np.float32),
                            actions=np.zeros((len(sources), 4), dtype=np.float32),
                            source_episode_ids=np.array(sources),
                            configuration_ids=np.array([f"{source}_{split}_000" for source in sources]))
    model = Policy(hidden_size=8, feature_indices=range(9, 30))
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(checkpoint, model, Normalizer(np.zeros(30, np.float32), np.ones(30, np.float32)),
                    {"binding": binding_for_manifest(manifest_path), "training_sources": ["E01", "E05"]})
    return dataset, checkpoint, frozen


def test_collection_preserves_bound_manifest_validation_bytes_and_appends_training_rows(tmp_path):
    from ring_learning.integrity import sha256_file, validate_split_arrays, verify_binding
    from ring_learning.policy import load_checkpoint
    dagger = dagger_module()
    assert hasattr(dagger, "collect"), "Checkpoint-bound dataset collector is missing"
    dataset, checkpoint, frozen = small_bound_dataset(tmp_path)
    output = tmp_path / "dagger_dataset"
    report = dagger.collect(dataset, checkpoint, output, per_source=1, beta=0.,
                            seed=20261016, max_steps=3)
    assert (output / "configurations.json").read_bytes() == (dataset / "configurations.json").read_bytes()
    assert (output / "val.npz").read_bytes() == (dataset / "val.npz").read_bytes()
    assert report["correction_row_count"] == 6
    assert report["combined_train_rows"] == 8
    assert report["baseline_checkpoint_sha256"] == sha256_file(checkpoint)
    assert report["source_episode_ids"] == ["E01", "E05"]
    assert report["all_label_states_train_only"] is True
    assert report["beta"] == 0.
    assert report["seed"] == 20261016
    assert len(report["rollouts"]) == 2
    assert all(row["split"] == "train" and row["steps"] == 3 for row in report["rollouts"])
    assert json.loads((output / "dagger_collection.json").read_text(encoding="utf-8")) == report
    with np.load(output / "train.npz", allow_pickle=False) as combined:
        assert validate_split_arrays(frozen, "train", combined) == ["E01", "E05"]
        assert combined["observations"].shape == (8, 30)
        assert combined["actions"].shape == (8, 4)
        with np.load(dataset / "train.npz", allow_pickle=False) as original:
            for key in original.files:
                np.testing.assert_array_equal(combined[key][:2], original[key])
    _, _, metadata = load_checkpoint(checkpoint)
    assert verify_binding(metadata["binding"], output / "configurations.json")["environment_changed"] is False


def test_collection_rejects_changed_manifest_before_writing_output(tmp_path):
    dagger = dagger_module()
    assert hasattr(dagger, "collect"), "Checkpoint-bound dataset collector is missing"
    dataset, checkpoint, frozen = small_bound_dataset(tmp_path)
    manifest_path = dataset / "configurations.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    output = tmp_path / "dagger_dataset"
    with pytest.raises(ValueError, match="manifest"):
        dagger.collect(dataset, checkpoint, output, per_source=1, max_steps=3)
    assert not output.exists()


def test_collection_rejects_wrong_training_array_source_before_writing_output(tmp_path):
    dagger = dagger_module()
    assert hasattr(dagger, "collect"), "Checkpoint-bound dataset collector is missing"
    dataset, checkpoint, frozen = small_bound_dataset(tmp_path)
    with np.load(dataset / "train.npz", allow_pickle=False) as original:
        arrays = {key: original[key].copy() for key in original.files}
    arrays["source_episode_ids"][0] = "E02"
    np.savez_compressed(dataset / "train.npz", **arrays)
    output = tmp_path / "dagger_dataset"
    with pytest.raises(ValueError, match="pair"):
        dagger.collect(dataset, checkpoint, output, per_source=1, max_steps=3)
    assert not output.exists()


def test_collection_never_overwrites_existing_dataset(tmp_path):
    dagger = dagger_module()
    assert hasattr(dagger, "collect"), "Checkpoint-bound dataset collector is missing"
    dataset, checkpoint, frozen = small_bound_dataset(tmp_path)
    before = {name: (dataset / name).read_bytes() for name in ("configurations.json", "train.npz", "val.npz")}
    with pytest.raises(FileExistsError):
        dagger.collect(dataset, checkpoint, dataset, per_source=1, max_steps=3)
    assert all((dataset / name).read_bytes() == content for name, content in before.items())


def test_probability_mixture_is_reproducible_and_executes_both_controllers():
    dagger = dagger_module()
    frozen = manifest()
    selected = dagger.select_training_configurations(frozen, per_source=3, seed=31)
    student_action = np.array([.3, -.2, .1, -1], dtype=np.float32)
    first_env, second_env = CounterEnv(), CounterEnv()
    first, logs = dagger.collect_corrections(first_env, frozen, selected,
                                             lambda obs: student_action, correction_label,
                                             beta=.25, seed=31)
    second, second_logs = dagger.collect_corrections(second_env, frozen, selected,
                                                     lambda obs: student_action, correction_label,
                                                     beta=.25, seed=31)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert logs == second_logs
    interventions = sum(row["teacher_interventions"] for row in logs)
    assert 0 < interventions < len(first["observations"])
    for executed, label in zip(first_env.executed_actions, first["actions"]):
        np.testing.assert_array_equal(executed, label if executed[3] > 0 else student_action)
    assert sum(action[3] > 0 for action in first_env.executed_actions) == interventions
    np.testing.assert_allclose(first["actions"][:, 0], first["observations"][:, 0] / 10)


@pytest.mark.parametrize("kwargs", [{"observation_noise": -.001}, {"observation_noise": np.inf},
                                    {"max_steps": 0}, {"max_steps": 1.5}])
def test_rollout_rejects_invalid_noise_or_horizon_before_reset(kwargs):
    dagger = dagger_module()
    frozen, env = manifest(), CounterEnv()
    with pytest.raises(ValueError, match="observation_noise|max_steps"):
        dagger.collect_corrections(env, frozen, frozen["configurations"]["train"][:1],
                                   lambda obs: np.zeros(4), correction_label, **kwargs)
    assert env.reset_positions == []


@pytest.mark.parametrize("bad_action", [np.zeros(3), np.full(4, np.nan)])
def test_rollout_rejects_invalid_student_actions(bad_action):
    dagger = dagger_module()
    frozen, env = manifest(), CounterEnv()
    with pytest.raises(ValueError, match="Student"):
        dagger.collect_corrections(env, frozen, frozen["configurations"]["train"][:1],
                                   lambda obs: bad_action, correction_label, beta=0)
    assert env.executed_actions == []


@pytest.mark.parametrize("bad_action", [np.zeros(3), np.full(4, np.inf), np.full(4, 1.1)])
def test_rollout_rejects_invalid_teacher_labels(bad_action):
    dagger = dagger_module()
    frozen, env = manifest(), CounterEnv()
    with pytest.raises(ValueError, match="Teacher"):
        dagger.collect_corrections(env, frozen, frozen["configurations"]["train"][:1],
                                   lambda obs: np.zeros(4), lambda env: bad_action, beta=0)
    assert env.executed_actions == []


def test_train_collection_reports_progress_for_each_configuration(capsys):
    dagger = dagger_module()
    frozen, env = manifest(), CounterEnv()
    selected = dagger.select_training_configurations(frozen, per_source=1)
    dagger.collect_corrections(env, frozen, selected, lambda obs: np.zeros(4), correction_label, beta=0)
    progress = capsys.readouterr().out
    assert "train/DAgger 1/2" in progress
    assert "train/DAgger 2/2" in progress


def test_cli_exposes_requested_collection_flags():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "ring_learning.dagger", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for flag in ("--dataset", "--checkpoint", "--output", "--per-source", "--beta", "--seed"):
        assert flag in result.stdout
