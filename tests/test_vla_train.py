import hashlib
import json

import numpy as np
import pytest
import torch


def _fixture_dataset(tmp_path):
    from ring_learning.vla_policy import N_WAYPOINTS
    manifest = {"schema_version": 1, "task_names": ["inside", "left", "right"],
                "split_sources": {"train": ["E01"], "val": ["E05"]},
                "configurations": {}, "files": {}}
    for split, source in [("train", "E01"), ("val", "E05")]:
        identifier = split+"_000"
        manifest["configurations"][split] = [{"configuration_id": identifier, "source_episode_id": source}]
        masks = np.zeros((1, 2, 256, 256), dtype=bool)
        masks[0, 0, 40:45, 60:65] = True
        masks[0, 1, 130:135, 140:145] = True
        path = tmp_path/f"{split}.npz"
        np.savez_compressed(path, images=np.zeros((1, 256, 256, 3), np.uint8), masks=masks,
            coordinates=np.array([[.33, .12, .48, -.02]], np.float32),
            waypoints=np.zeros((1, 3, N_WAYPOINTS, 4), np.float32),
            source_episode_ids=np.array([source]), configuration_ids=np.array([identifier]))
        manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_loader_binds_train_val_hashes_and_source_membership(tmp_path):
    from ring_learning.vla_train import load_development_dataset
    _fixture_dataset(tmp_path)
    train, val, manifest, binding = load_development_dataset(tmp_path)
    assert train["source_episode_ids"].tolist() == ["E01"]
    assert val["source_episode_ids"].tolist() == ["E05"]
    assert len(binding["dataset_manifest_sha256"]) == 64
    assert len(binding["train_npz_sha256"]) == 64


def test_loader_rejects_cross_split_sources_before_training(tmp_path):
    from ring_learning.vla_train import load_development_dataset
    manifest = _fixture_dataset(tmp_path)
    manifest["split_sources"]["val"] = ["E01"]
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source|leakage"):
        load_development_dataset(tmp_path)


def test_loader_rejects_replaced_npz_even_if_its_arrays_are_loadable(tmp_path):
    from ring_learning.vla_train import load_development_dataset
    _fixture_dataset(tmp_path)
    with (tmp_path/"train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash|SHA"):
        load_development_dataset(tmp_path)


def test_decoder_prefit_recovers_affine_grasp_and_three_different_language_goals():
    from ring_learning.vla_policy import scene_feature_basis
    from ring_learning.vla_train import fit_decoder_matrices
    rng = np.random.default_rng(19)
    coordinates = rng.uniform([.30, -.20, .40, -.10], [.60, .20, .55, .10], (50, 4)).astype(np.float32)
    features = scene_feature_basis(torch.from_numpy(coordinates)).numpy()
    matrices = np.zeros((3, 23, 4, 10), np.float32)
    matrices[:, :, 0, 2] = 1.
    matrices[:, :, 1, 3] = 1.
    matrices[:, :, 2, 4] = .1043
    matrices[:, :, 3, 4] = 1.
    matrices[1, :, 0, 4] = -.06
    matrices[2, :, 0, 4] = .06
    expected = np.einsum("tkaf,nf->ntka", matrices, features)
    fitted = fit_decoder_matrices(coordinates, expected, np.array(["E01"]*25+["E02"]*25))
    predicted = np.einsum("tkaf,nf->ntka", fitted, features)
    np.testing.assert_allclose(predicted, expected, atol=2e-6, rtol=0.)
    assert predicted[0, 1, 0, 0] < predicted[0, 0, 0, 0] < predicted[0, 2, 0, 0]


def test_validation_metrics_separate_grasp_goal_and_transfer_error():
    from ring_learning.vla_train import action_error_metrics
    labels = np.zeros((2, 3, 23, 4), np.float32)
    prediction = labels.copy()
    prediction[:, :, 1, 0] = .001
    prediction[:, :, 19, 1] = .002
    prediction[:, :, 10, 0] = .050
    scores = action_error_metrics(prediction, labels)
    assert scores["grasp_xy_mean_mm"] == pytest.approx(1.)
    assert scores["goal_xy_mean_mm"] == pytest.approx(2.)
    assert scores["critical_xy_max_mm"] == pytest.approx(2.)
    assert scores["all_xyz_mean_mm"] > scores["grasp_xy_mean_mm"]


def test_source_weights_do_not_count_many_augmentations_as_more_human_sources():
    from ring_learning.vla_train import source_uniform_weights
    weights = source_uniform_weights(np.array(["E01", "E02", "E02", "E02"]))
    assert weights[0] == pytest.approx(weights[1:].sum())
    assert weights.mean() == pytest.approx(1.)


def test_training_only_readout_refinement_can_remove_tiny_text_fit_errors():
    from ring_learning.vla_policy import TinyVLA, encode_text, INSTRUCTION_GROUPS
    from ring_learning.vla_train import refine_language_readout
    torch.manual_seed(8)
    model = TinyVLA(n_waypoints=1)
    phrases = [phrase for group in INSTRUCTION_GROUPS.values() for phrase in group]
    tokens = encode_text(phrases)
    targets = torch.zeros(9, 40)
    targets[:3, 0] = .49
    targets[3:6, 0] = .43
    targets[6:, 0] = .55
    report = refine_language_readout(model, tokens, targets)
    actual = model.coefficient_head(model.text_features(tokens))
    torch.testing.assert_close(actual, targets, atol=1e-5, rtol=0.)
    assert report["maximum_coefficient_error"] < 1e-5
