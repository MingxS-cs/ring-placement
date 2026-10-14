"""Train a task-specific hierarchical VLA on video-derived simulation chunks.

Only the frozen development train/validation NPZ files are opened. Segmentation
and true scene coordinates supervise training; deployment must recover geometry
from RGB through the learned masks. The inherited teacher supplies synthetic
robot action labels, not measured human joint/gripper/depth demonstrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .vla_policy import (INSTRUCTION_GROUPS, N_WAYPOINTS,
                         PRIMARY_INSTRUCTIONS, TinyVLA, encode_text,
                         policy_source_sha256, save_vla_policy, scene_feature_basis)


TASK_NAMES = ("inside", "left", "right")
CRITICAL_INDICES = (0, 1, 2, 3, 19, 20, 21, 22)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_source_sha256():
    text = Path(__file__).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_arrays(arrays, split, manifest):
    required = {"images", "masks", "coordinates", "waypoints", "source_episode_ids", "configuration_ids"}
    if not required.issubset(arrays):
        raise ValueError(f"{split} NPZ is missing required arrays")
    images, masks = arrays["images"], arrays["masks"]
    n = len(images)
    if n < 1 or images.shape != (n, 256, 256, 3) or images.dtype != np.uint8:
        raise ValueError(f"{split} images must be nonempty uint8 N x 256 x 256 x 3")
    if masks.shape != (n, 2, 256, 256) or masks.dtype != bool:
        raise ValueError(f"{split} masks must be bool N x 2 x 256 x 256")
    if np.any(masks.sum(axis=(2, 3)) == 0):
        raise ValueError(f"{split} samples must contain both visible object masks")
    for name, shape in (("coordinates", (n, 4)), ("waypoints", (n, 3, N_WAYPOINTS, 4))):
        values = arrays[name]
        if values.shape != shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{split} {name} shape or finite numeric values are invalid")
    if np.any(np.abs(arrays["waypoints"][..., 3]) > 1.+1e-6):
        raise ValueError(f"{split} gripper labels must be normalized to [-1, 1]")
    for name in ("source_episode_ids", "configuration_ids"):
        if arrays[name].shape != (n,) or arrays[name].dtype.kind != "U":
            raise ValueError(f"{split} {name} must be an N-vector of strings")
    sources = set(arrays["source_episode_ids"].tolist())
    if sources != set(manifest["split_sources"][split]):
        raise ValueError(f"{split} source membership differs from the frozen manifest")
    pairs = {(item["configuration_id"], item["source_episode_id"])
             for item in manifest["configurations"][split]}
    actual = list(zip(arrays["configuration_ids"].tolist(), arrays["source_episode_ids"].tolist()))
    if len(set(actual)) != n or set(actual) != pairs:
        raise ValueError(f"{split} configuration/source lineage differs from the frozen manifest")


def load_development_dataset(directory):
    """Open only train/val, checking exact bytes and source/configuration isolation."""
    directory = Path(directory)
    path = directory/"manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("task_names") != list(TASK_NAMES):
        raise ValueError("Unsupported VLA dataset schema or task order")
    if not all(split in manifest.get("split_sources", {}) and
               split in manifest.get("configurations", {}) for split in ("train", "val")):
        raise ValueError("Dataset manifest must specify train and validation source membership")
    train_sources, val_sources = (manifest["split_sources"][split] for split in ("train", "val"))
    if (not train_sources or not val_sources or set(train_sources) & set(val_sources)
            or len(set(train_sources)) != len(train_sources) or len(set(val_sources)) != len(val_sources)):
        raise ValueError("Dataset source leakage or duplicate source membership")
    configuration_ids = []
    for split in ("train", "val"):
        for item in manifest["configurations"][split]:
            if (not isinstance(item.get("configuration_id"), str)
                    or item.get("source_episode_id") not in manifest["split_sources"][split]
                    or ("split" in item and item["split"] != split)):
                raise ValueError("Configuration/source membership is invalid")
            configuration_ids.append(item["configuration_id"])
    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("Configuration leakage or duplicate configuration identifiers")
    loaded, binding = {}, {"dataset_manifest_sha256": sha256_file(path)}
    for split in ("train", "val"):
        filename = f"{split}.npz"
        expected = manifest.get("files", {}).get(filename, {}).get("sha256")
        actual_hash = sha256_file(directory/filename)
        if expected != actual_hash:
            raise ValueError(f"{filename} SHA256 hash differs from the frozen dataset manifest")
        with np.load(directory/filename, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        _validate_arrays(arrays, split, manifest)
        loaded[split] = arrays
        binding[f"{split}_npz_sha256"] = actual_hash
    return loaded["train"], loaded["val"], manifest, binding


def source_uniform_weights(source_ids):
    values = np.asarray(source_ids)
    if values.ndim != 1 or len(values) < 1 or values.dtype.kind != "U":
        raise ValueError("Source IDs must be a nonempty string vector")
    sources, counts = np.unique(values, return_counts=True)
    inverse = {source: len(values)/(len(sources)*count) for source, count in zip(sources, counts)}
    return np.array([inverse[source] for source in values], dtype=np.float64)


def fit_decoder_matrices(coordinates, waypoints, source_ids):
    """Training-only weighted least-squares target for the learned text decoder."""
    coordinates = torch.as_tensor(np.asarray(coordinates), dtype=torch.float64)
    features = scene_feature_basis(coordinates).numpy()
    labels = np.asarray(waypoints, dtype=np.float64)
    if labels.shape != (len(features), 3, N_WAYPOINTS, 4) or not np.isfinite(labels).all():
        raise ValueError("Action-chunk labels must be finite N x 3 x 23 x 4")
    weights = source_uniform_weights(source_ids)
    if len(weights) != len(features):
        raise ValueError("Source weighting length differs from the geometry labels")
    weighted = np.sqrt(weights)[:, None]
    matrix = []
    for task in range(3):
        coefficients, _, _, _ = np.linalg.lstsq(features*weighted,
            labels[:, task].reshape(len(features), -1)*weighted, rcond=None)
        matrix.append(coefficients.T.reshape(N_WAYPOINTS, 4, 10))
    return np.asarray(matrix, dtype=np.float32)


def action_error_metrics(prediction, labels):
    prediction, labels = np.asarray(prediction, float), np.asarray(labels, float)
    if prediction.shape != labels.shape or prediction.ndim != 4 or prediction.shape[1:] != (3, 23, 4):
        raise ValueError("Predictions and labels must have equal N x 3 x 23 x 4 shapes")
    delta = prediction-labels
    xy = np.linalg.norm(delta[..., :2], axis=-1)*1000
    xyz = np.linalg.norm(delta[..., :3], axis=-1)*1000
    return dict(grasp_xy_mean_mm=float(xy[:, :, 1].mean()),
                grasp_xy_max_mm=float(xy[:, :, 1].max()),
                goal_xy_mean_mm=float(xy[:, :, 19].mean()),
                goal_xy_max_mm=float(xy[:, :, 19].max()),
                critical_xy_max_mm=float(xy[:, :, CRITICAL_INDICES].max()),
                critical_xy_mean_mm=float(xy[:, :, CRITICAL_INDICES].mean()),
                all_xyz_mean_mm=float(xyz.mean()),
                all_xyz_max_mm=float(xyz.max()),
                gripper_max_error=float(np.abs(delta[..., 3]).max()))


def _evaluate(model, arrays, device, batch_size):
    predictions, features, all_logits = [], [], []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(arrays["images"]), batch_size):
            images = torch.from_numpy(arrays["images"][begin:begin+batch_size]).to(device).permute(0, 3, 1, 2).float()/255.
            feature, logits = model.scene_features(images)
            chunks = []
            for name in TASK_NAMES:
                tokens = encode_text([PRIMARY_INSTRUCTIONS[name]]*len(images), device=device)
                chunks.append(model.predict_from_scene_features(feature, tokens).cpu().numpy())
            predictions.append(np.stack(chunks, axis=1))
            features.append(feature[:, :4].cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    predicted = np.concatenate(predictions)
    detected = np.concatenate(features)
    error = (detected-arrays["coordinates"]).reshape(-1, 2, 2)
    norms = np.linalg.norm(error, axis=-1)*1000
    logits = np.concatenate(all_logits)
    mask_predictions = logits > 0
    masks = arrays["masks"]
    intersection = (mask_predictions & masks).sum(axis=(2, 3))
    union = (mask_predictions | masks).sum(axis=(2, 3))
    scores = action_error_metrics(predicted, arrays["waypoints"])
    scores.update(ring_xy_mean_mm=float(norms[:, 0].mean()), ring_xy_max_mm=float(norms[:, 0].max()),
                  bearing_xy_mean_mm=float(norms[:, 1].mean()), bearing_xy_max_mm=float(norms[:, 1].max()),
                  ring_mask_iou=float((intersection[:, 0]/union[:, 0]).mean()),
                  bearing_mask_iou=float((intersection[:, 1]/union[:, 1]).mean()))
    return scores


def refine_language_readout(model, tokens, targets):
    """Fit only the final neural linear readout on the nine training phrases.

    Learned word/hidden features remain inputs, not intent IDs. Solving this
    small supervised regression removes optimizer residue that would add up
    across the ten metric geometry terms when emitting robot waypoints.
    """
    with torch.no_grad():
        hidden = model.coefficient_head[:-1](model.text_features(tokens)).cpu().numpy().astype(np.float64)
        design = np.column_stack((hidden, np.ones(len(hidden))))
        target = targets.detach().cpu().numpy().astype(np.float64)
        coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-7)
        readout = model.coefficient_head[-1]
        readout.weight.copy_(torch.as_tensor(coefficients[:-1].T, dtype=readout.weight.dtype, device=readout.weight.device))
        readout.bias.copy_(torch.as_tensor(coefficients[-1], dtype=readout.bias.dtype, device=readout.bias.device))
        maximum = float((model.coefficient_head(model.text_features(tokens))-targets).abs().max())
    return dict(maximum_coefficient_error=maximum, design_rank=int(rank),
                largest_singular_value=float(singular[0]), smallest_singular_value=float(singular[-1]),
                maximum_absolute_readout_coefficient=float(np.abs(coefficients).max()),
                method="float64 training-phrase least squares on learned hidden features")


def _fit_language_decoder(model, matrices, device, steps, seed):
    phrases, targets = [], []
    for task, name in enumerate(TASK_NAMES):
        for phrase in INSTRUCTION_GROUPS[name]:
            phrases.append(phrase)
            targets.append(matrices[task])
    tokens = encode_text(phrases, device=device)
    target = torch.as_tensor(np.asarray(targets), device=device).flatten(1)
    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    final_loss = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        coefficients = model.coefficient_head(model.text_features(tokens))
        loss = F.mse_loss(coefficients, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach())
    with torch.inference_mode():
        maximum = float((model.coefficient_head(model.text_features(tokens))-target).abs().max())
    refinement = refine_language_readout(model, tokens, target)
    return dict(steps=steps, loss=final_loss, maximum_coefficient_error=maximum,
                readout_refinement=refinement, all_trained_phrases=phrases,
                method="weighted scene-feature least squares, learned word-network coefficient supervision and training-only linear-readout refinement")


def train_vla(dataset_directory, output_directory, *, device="cuda", seed=20261025,
              epochs=60, batch_size=8, decoder_steps=2000, learning_rate=.003,
              centre_loss_weight=.02, brightness_jitter=.08):
    if epochs < 1 or batch_size < 1 or decoder_steps < 1:
        raise ValueError("Training counts must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but this environment has no CUDA PyTorch")
    start_time = time.perf_counter()
    train, val, manifest, binding = load_development_dataset(dataset_directory)
    model = TinyVLA().to(device)
    matrices = fit_decoder_matrices(train["coordinates"], train["waypoints"], train["source_episode_ids"])
    decoder_report = _fit_language_decoder(model, matrices, device, decoder_steps, seed)
    for parameter in model.decoder_parameters():
        parameter.requires_grad_(False)
    train_images = torch.from_numpy(train["images"]).permute(0, 3, 1, 2).float()/255.
    train_masks = torch.from_numpy(train["masks"]).float()
    train_coordinates = torch.from_numpy(train["coordinates"]).float()
    weights = torch.from_numpy(source_uniform_weights(train["source_episode_ids"])).float()
    positive = train_masks.sum(dim=(0, 2, 3))
    pixels = len(train_masks)*256*256
    positive_weight = ((pixels-positive)/positive).to(device).view(1, 2, 1, 1)
    optimizer = torch.optim.Adam(model.vision.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*.01)
    generator = torch.Generator().manual_seed(seed+1)
    history, best, best_state = [], None, None
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_images), generator=generator)
        loss_sum, batch_count = 0., 0
        for indices in permutation.split(batch_size):
            images = train_images[indices].to(device)
            if brightness_jitter:
                factor = 1+(torch.rand(len(images), 1, 1, 1, device=device)*2-1)*brightness_jitter
                images = (images*factor).clamp(0, 1)
            masks = train_masks[indices].to(device)
            coordinates = train_coordinates[indices].to(device)
            sample_weights = weights[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            features, logits = model.scene_features(images)
            mask_loss = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=positive_weight, reduction="none").mean(dim=(1, 2, 3))
            centre_loss = ((features[:, :4]-coordinates)*1000).square().mean(dim=1)
            ramp = min(1., max(0., (epoch-3)/7))
            loss = ((mask_loss+centre_loss_weight*ramp*centre_loss)*sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vision.parameters(), 10.)
            optimizer.step()
            loss_sum += float(loss.detach()); batch_count += 1
        scheduler.step()
        metrics = _evaluate(model, val, device, batch_size)
        metrics.update(epoch=epoch+1, training_loss=loss_sum/batch_count)
        history.append(metrics)
        selection_score = (metrics["critical_xy_max_mm"], metrics["critical_xy_mean_mm"])
        if best is None or selection_score < best:
            best = selection_score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(json.dumps(dict(epoch=epoch+1, grasp_max_mm=metrics["grasp_xy_max_mm"],
                goal_max_mm=metrics["goal_xy_max_mm"], train_loss=metrics["training_loss"])), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    selected_index = min(range(len(history)), key=lambda i: (history[i]["critical_xy_max_mm"], history[i]["critical_xy_mean_mm"]))
    final_validation = _evaluate(model, val, device, batch_size)
    final_training = _evaluate(model, train, device, batch_size)
    metadata = dict(**binding, training_sources=manifest["split_sources"]["train"],
        validation_sources=manifest["split_sources"]["val"], parameter_count=model.parameter_count,
        training_sample_count=len(train_images), validation_sample_count=len(val["images"]),
        implementation_sources=dict(vla_policy=policy_source_sha256(), vla_train=training_source_sha256()),
        data_manifest=manifest, method="hierarchical task-specific VLA with learned visual masks and word-conditioned 23-waypoint XYZ/gripper chunk",
        action_label_origin="video-derived V4 planar trajectory with manually designed height/gripper stages; robot chunks are simulation-generated",
        runtime_object_coordinates="predicted from learned RGB masks only; no simulator object state input",
        language_scope="nine registered English phrases covering inside/left/right; no open-vocabulary claims",
        selection=dict(split="validation only", selected_epoch=selected_index+1,
                       criterion="minimum maximum critical-waypoint XY error, then mean critical XY error",
                       trained_epochs=epochs, blind_test_opened=False),
        hyperparameters=dict(seed=seed, batch_size=batch_size, epochs=epochs, decoder_steps=decoder_steps,
            learning_rate=learning_rate, centre_loss_weight=centre_loss_weight, brightness_jitter=brightness_jitter,
            learning_rate_schedule="cosine decay to 1 percent of initial vision learning rate; language cosine decay to 1e-5",
            centroid="squared sigmoid foreground mass", source_weighting="equal total contribution per original human source"),
        decoder_prefit=decoder_report, training_metrics=final_training, validation_metrics=final_validation,
        environment=dict(torch=str(torch.__version__), numpy=str(np.__version__), device=device,
                         gpu_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                         deterministic_algorithms=True), wall_time_seconds=time.perf_counter()-start_time)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory/"policy.pt"
    save_vla_policy(path, model, metadata)
    report = dict(frozen_checkpoint_sha256=sha256_file(path), metadata=metadata, validation_history=history)
    (output_directory/"training.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    np.save(output_directory/"decoder_training_coefficients.npy", matrices, allow_pickle=False)
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/vla_v5/dataset")
    parser.add_argument("--out", default="outputs/vla_v5/model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder-steps", type=int, default=2000)
    args = parser.parse_args()
    _, report = train_vla(args.dataset, args.out, device=args.device, epochs=args.epochs,
                          batch_size=args.batch_size, decoder_steps=args.decoder_steps)
    print(json.dumps(dict(checkpoint_sha256=report["frozen_checkpoint_sha256"],
                         validation=report["metadata"]["validation_metrics"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
