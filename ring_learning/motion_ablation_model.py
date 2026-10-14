"""Training-only removal of video-derived transfer curvature from the V5 VLA.

Only the final readout rows for intermediate transfer XY are refitted. The
recorded source endpoints remain in both conditions: this is a partial motion
ablation, not an experiment removing every contribution of the collected video.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .vla_policy import (FEATURE_DIM, INSTRUCTION_GROUPS, N_WAYPOINTS,
                         TinyVLA, encode_text, scene_feature_basis)
from .vla_train import fit_decoder_matrices, refine_language_readout


TRANSFER_INDICES = tuple(range(4, 19))
TRANSFER_PROGRESS = np.arange(1, 16, dtype=np.float64)/16.


def linearize_transfer(chunks):
    """Copy to float32 and replace only waypoints 4..18 XY with the chord.

    For the frozen float32 training arrays every other label retains its exact
    bytes. Other finite numeric inputs are converted to float32 before the
    intervention; a conversion that cannot represent them finitely is rejected.
    """
    values = np.asarray(chunks)
    if values.ndim != 4 or values.shape[1:] != (3, N_WAYPOINTS, 4) or len(values) < 1:
        raise ValueError("Action chunks must have nonempty N x 3 x 23 x 4 shape")
    if (values.dtype.kind not in "fiu" or not np.isfinite(values).all()
            or np.any(np.abs(values) > np.finfo(np.float32).max)):
        raise ValueError("Action chunks must contain finite float32-representable numeric values")
    result = np.array(values, dtype=np.float32, copy=True)
    progress = TRANSFER_PROGRESS[None, None, :, None]
    result[:, :, 4:19, :2] = ((1-progress)*result[:, :, 3:4, :2]
                              + progress*result[:, :, 19:20, :2]).astype(np.float32)
    return result


def changed_readout_rows():
    """Rows of the flattened 23 x 4 x 10 final coefficient readout."""
    return np.array([(index*4+axis)*FEATURE_DIM+feature
        for index in TRANSFER_INDICES for axis in (0, 1)
        for feature in range(FEATURE_DIM)], dtype=np.int64)


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _validate_training_arrays(train_arrays):
    try:
        coordinates = np.asarray(train_arrays["coordinates"])
        source_ids = np.asarray(train_arrays["source_episode_ids"])
        chunks = np.asarray(train_arrays["waypoints"])
    except (KeyError, TypeError) as error:
        raise ValueError("The training arrays must supply coordinates, source_episode_ids and waypoints") from error
    if (coordinates.ndim != 2 or coordinates.shape[1] != 4
            or coordinates.dtype.kind not in "fiu" or not np.isfinite(coordinates).all()):
        raise ValueError("Training geometry must contain finite N x 4 coordinates")
    straight = linearize_transfer(chunks)
    n = len(straight)
    if len(coordinates) != n or source_ids.shape != (n,) or source_ids.dtype.kind != "U":
        raise ValueError("Training coordinates, sources and action chunks must have matching rows")
    if any(not source for source in source_ids.tolist()):
        raise ValueError("Training source identifiers must be nonempty")
    return coordinates, source_ids, chunks, straight


def fit_straight_model(base_model, train_arrays):
    """Copy the frozen model and refit only intermediate transfer XY rows.

    No dataset loader, held-out data or physical outcome is accessed. The caller
    must establish the supplied arrays' TRAIN membership from the sealed dataset
    manifest. All nine already-trained phrases participate in the readout fit.
    """
    if not isinstance(base_model, TinyVLA) or base_model.n_waypoints != N_WAYPOINTS:
        raise ValueError("The base model must be the 23-waypoint TinyVLA")
    coordinates, source_ids, curved_labels, straight_labels = _validate_training_arrays(train_arrays)
    original_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    fitted = copy.deepcopy(base_model)
    device = next(fitted.parameters()).device
    dtype = next(fitted.parameters()).dtype
    if dtype != torch.float32:
        raise ValueError("The frozen V5 base model must use float32 parameters")
    matrices = fit_decoder_matrices(coordinates, straight_labels, source_ids)
    phrases = [phrase for group in INSTRUCTION_GROUPS.values() for phrase in group]
    tokens = encode_text(phrases, device=device)
    coefficient_targets = torch.from_numpy(np.repeat(matrices, 3, axis=0).reshape(9, -1)).to(device)
    refinement = refine_language_readout(fitted, tokens, coefficient_targets)
    changed = changed_readout_rows()
    common = np.setdiff1d(np.arange(N_WAYPOINTS*4*FEATURE_DIM), changed)
    common_tensor = torch.as_tensor(common, dtype=torch.long, device=device)
    readout = fitted.coefficient_head[-1]
    with torch.no_grad():
        readout.weight[common_tensor] = base_model.coefficient_head[-1].weight[common_tensor]
        readout.bias[common_tensor] = base_model.coefficient_head[-1].bias[common_tensor]

    fitted_state = fitted.state_dict()
    readout_names = ("coefficient_head.2.weight", "coefficient_head.2.bias")
    unchanged_parameters_equal = all(_tensor_bytes(tensor) == _tensor_bytes(original_state[name])
        for name, tensor in fitted_state.items() if name not in readout_names)
    common_readout_rows_equal = all(_tensor_bytes(fitted_state[name][common_tensor]) ==
        _tensor_bytes(original_state[name][common_tensor]) for name in readout_names)
    base_model_unchanged = all(_tensor_bytes(tensor) == _tensor_bytes(original_state[name])
        for name, tensor in base_model.state_dict().items())
    if not (unchanged_parameters_equal and common_readout_rows_equal and base_model_unchanged):
        raise RuntimeError("A supposedly shared model parameter changed during the transfer intervention")

    features = scene_feature_basis(torch.as_tensor(coordinates, dtype=dtype, device=device))
    common_outputs = torch.ones((N_WAYPOINTS, 4), dtype=torch.bool, device=device)
    common_outputs[4:19, :2] = False
    max_curve_residual, max_fit_error, max_common_difference = 0., 0., 0.
    max_transfer_difference, common_outputs_equal = 0., True
    fitted.eval()
    with torch.inference_mode():
        for task, group in enumerate(INSTRUCTION_GROUPS.values()):
            for phrase in group:
                batch_tokens = encode_text([phrase]*len(features), device=device)
                reference = base_model.predict_from_scene_features(features, batch_tokens)
                prediction = fitted.predict_from_scene_features(features, batch_tokens)
                progress = torch.as_tensor(TRANSFER_PROGRESS, dtype=dtype, device=device).view(1, 15, 1)
                chord = (1-progress)*prediction[:, 3:4, :2]+progress*prediction[:, 19:20, :2]
                max_curve_residual = max(max_curve_residual, float(torch.linalg.vector_norm(
                    prediction[:, 4:19, :2]-chord, dim=-1).max()))
                label = torch.as_tensor(straight_labels[:, task, 4:19, :2], device=device)
                max_fit_error = max(max_fit_error, float(torch.linalg.vector_norm(
                    prediction[:, 4:19, :2]-label, dim=-1).max()))
                common_difference = (prediction-reference)[:, common_outputs]
                max_common_difference = max(max_common_difference, float(common_difference.abs().max()))
                common_outputs_equal &= (_tensor_bytes(prediction[:, common_outputs]) ==
                                         _tensor_bytes(reference[:, common_outputs]))
                max_transfer_difference = max(max_transfer_difference, float(torch.linalg.vector_norm(
                    (prediction-reference)[:, 4:19, :2], dim=-1).max()))
    if not common_outputs_equal:
        raise RuntimeError("A supposedly shared predicted action changed during the transfer intervention")
    feature_rank = int(np.linalg.matrix_rank(scene_feature_basis(
        torch.as_tensor(coordinates, dtype=torch.float64)).numpy()))
    label_curve_difference = np.linalg.norm(
        straight_labels[:, :, 4:19, :2]-curved_labels[:, :, 4:19, :2], axis=-1)
    actual_changed = [int(row) for row in changed
        if _tensor_bytes(fitted_state[readout_names[0]][row]) != _tensor_bytes(original_state[readout_names[0]][row])
        or _tensor_bytes(fitted_state[readout_names[1]][row:row+1]) != _tensor_bytes(original_state[readout_names[1]][row:row+1])]
    report = dict(method="TRAIN-only video-curvature removal; recorded endpoint geometry retained",
        training_sample_count=len(coordinates), training_sources=sorted(set(source_ids.tolist())),
        all_training_phrases=phrases, transfer_waypoint_indices=list(TRANSFER_INDICES),
        transfer_progress=TRANSFER_PROGRESS.tolist(), changed_row_indices=changed.tolist(),
        actual_changed_row_indices=actual_changed, shared_readout_row_count=len(common),
        base_model_unchanged=base_model_unchanged, unchanged_parameters_equal=unchanged_parameters_equal,
        common_readout_rows_equal=common_readout_rows_equal, common_predicted_outputs_equal=common_outputs_equal,
        feature_design_rank=feature_rank, feature_design_columns=FEATURE_DIM,
        language_design_rank=refinement["design_rank"], language_design_columns=33,
        readout_refinement=refinement,
        train_maximum_straight_residual_m=max_curve_residual,
        train_maximum_transfer_fit_error_m=max_fit_error,
        train_maximum_common_output_difference=max_common_difference,
        train_maximum_video_straight_transfer_difference_m=max_transfer_difference,
        train_maximum_removed_label_curvature_m=float(label_curve_difference.max()),
        train_mean_removed_label_curvature_m=float(label_curve_difference.mean()),
        parameter_count=fitted.parameter_count, held_out_data_used=False)
    return fitted, report
