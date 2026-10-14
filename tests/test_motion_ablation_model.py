import numpy as np
import pytest
import torch


def _bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def test_label_intervention_copies_data_and_changes_only_transfer_xy():
    from ring_learning.motion_ablation_model import linearize_transfer
    chunks = np.random.default_rng(2).normal(size=(4, 3, 23, 4)).astype(np.float32)
    original = chunks.copy()
    result = linearize_transfer(chunks)
    assert result.dtype == np.float32 and not np.shares_memory(result, chunks)
    assert chunks.tobytes() == original.tobytes()
    preserved = np.ones(chunks.shape, bool)
    preserved[:, :, 4:19, :2] = False
    assert result[preserved].tobytes() == chunks[preserved].tobytes()
    progress = np.arange(1, 16, dtype=np.float64)/16.
    expected = ((1-progress)[None, None, :, None]*chunks[:, :, 3:4, :2]
                + progress[None, None, :, None]*chunks[:, :, 19:20, :2]).astype(np.float32)
    np.testing.assert_array_equal(result[:, :, 4:19, :2], expected)


@pytest.mark.parametrize("chunks", [np.zeros((2, 23, 4)), np.zeros((0, 3, 23, 4)),
    np.zeros((2, 3, 22, 4)), np.full((2, 3, 23, 4), np.nan),
    np.full((2, 3, 23, 4), np.inf), np.full((2, 3, 23, 4), 1e100)])
def test_invalid_action_arrays_are_rejected(chunks):
    from ring_learning.motion_ablation_model import linearize_transfer
    with pytest.raises(ValueError, match="shape|finite|float32|nonempty"):
        linearize_transfer(chunks)


def _training_fixture():
    from ring_learning.vla_policy import TinyVLA, INSTRUCTION_GROUPS, encode_text
    from ring_learning.vla_train import fit_decoder_matrices, refine_language_readout
    rng = np.random.default_rng(7)
    coordinates = rng.uniform([.32, -.18, .41, -.10], [.61, .18, .54, .10], size=(48, 4)).astype(np.float32)
    chunks = np.zeros((48, 3, 23, 4), np.float32)
    for task, offset in enumerate((0., -.06, .06)):
        start = coordinates[:, :2]+[0., .01125]
        goal = coordinates[:, 2:4]+[offset, .01125]
        chunks[:, task, :4, :2] = start[:, None]
        chunks[:, task, 19:, :2] = goal[:, None]
        chunks[:, task, :, 2] = .1643
        chunks[:, task, 1:3, 2] = .1043
        chunks[:, task, 20:22, 2] = .1043
        chunks[:, task, :, 3] = -1.
        chunks[:, task, [0, 21, 22], 3] = 1.
        for index, progress in enumerate(np.arange(1, 16)/16., start=4):
            chunks[:, task, index, :2] = (1-progress)*start+progress*goal+np.array([.009, -.006])*4*progress*(1-progress)
    sources = np.array(["E01"]*24+["E02"]*24)
    matrices = fit_decoder_matrices(coordinates, chunks, sources)
    torch.manual_seed(23)
    model = TinyVLA()
    phrases = [phrase for group in INSTRUCTION_GROUPS.values() for phrase in group]
    target = np.repeat(matrices, 3, axis=0).reshape(9, -1)
    refine_language_readout(model, encode_text(phrases), torch.from_numpy(target))
    return model, dict(coordinates=coordinates, waypoints=chunks, source_episode_ids=sources)


def test_fit_changes_only_designated_readout_rows_and_never_mutates_input_model(monkeypatch):
    from ring_learning.motion_ablation_model import fit_straight_model, changed_readout_rows
    import ring_learning.vla_train as old_train
    model, arrays = _training_fixture()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(old_train, "load_development_dataset", lambda *_: pytest.fail("must never load validation files"))
    fitted, report = fit_straight_model(model, arrays)
    assert fitted is not model
    changed = changed_readout_rows()
    common = np.setdiff1d(np.arange(23*4*10), changed)
    for name, value in model.state_dict().items():
        assert _bytes(value) == _bytes(original[name])
        if name not in ("coefficient_head.2.weight", "coefficient_head.2.bias"):
            assert _bytes(fitted.state_dict()[name]) == _bytes(original[name])
        else:
            assert _bytes(fitted.state_dict()[name][common]) == _bytes(original[name][common])
    assert report["changed_row_indices"] == changed.tolist()
    assert report["unchanged_parameters_equal"] and report["common_readout_rows_equal"]
    assert report["feature_design_rank"] == 9
    assert report["language_design_rank"] == 9
    assert len(changed) == 15*2*10


def test_fit_produces_straight_transfer_and_identical_grasp_height_gripper_outputs():
    from ring_learning.motion_ablation_model import fit_straight_model
    from ring_learning.vla_policy import encode_text, INSTRUCTION_GROUPS, scene_feature_basis
    model, arrays = _training_fixture()
    fitted, report = fit_straight_model(model, arrays)
    features = scene_feature_basis(torch.from_numpy(arrays["coordinates"]))
    common = np.ones((23, 4), bool)
    common[4:19, :2] = False
    with torch.inference_mode():
        for phrases in INSTRUCTION_GROUPS.values():
            for phrase in phrases:
                tokens = encode_text([phrase]*len(features))
                original = model.predict_from_scene_features(features, tokens)
                actual = fitted.predict_from_scene_features(features, tokens)
                assert _bytes(actual[:, common]) == _bytes(original[:, common])
                progress = torch.arange(1, 16, dtype=torch.float32).view(1, 15, 1)/16.
                linear = (1-progress)*actual[:, 3:4, :2]+progress*actual[:, 19:20, :2]
                torch.testing.assert_close(actual[:, 4:19, :2], linear, atol=1e-6, rtol=0.)
    assert report["train_maximum_straight_residual_m"] < 1e-6
    assert report["train_maximum_common_output_difference"] == 0.


def test_fit_rejects_missing_or_invalid_training_geometry():
    from ring_learning.motion_ablation_model import fit_straight_model
    model, arrays = _training_fixture()
    with pytest.raises(ValueError, match="training"):
        fit_straight_model(model, {"waypoints": arrays["waypoints"]})
    arrays["coordinates"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_straight_model(model, arrays)
