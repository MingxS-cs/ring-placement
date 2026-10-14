"""Behavior cloning with source-isolated validation and train-only statistics."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .dataset import Normalizer, noisy_observation, write_json
from .policy import Policy, save_checkpoint


def train(dataset_dir, output_dir, epochs=250, hidden=128, seed=20261014,
          observation_noise=.0002, learning_rate=.001):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "configurations.json").read_text(encoding="utf-8"))
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as data:
        train_obs, train_actions = data["observations"].copy(), data["actions"].copy()
        train_sources = set(data["source_episode_ids"].tolist())
    with np.load(dataset_dir / "val.npz", allow_pickle=False) as data:
        val_obs, val_actions = data["observations"].copy(), data["actions"].copy()
        val_sources = set(data["source_episode_ids"].tolist())
    if train_sources & val_sources:
        raise ValueError("Source episode leakage between train and validation")
    for split in ("test", "ood_test"):
        if train_sources & {e["source_episode_id"] for e in manifest["configurations"][split]}:
            raise ValueError("Evaluation source leaked into training")
    # Floors retain sensible scales for near-constant Cartesian/velocity features.
    floors = np.array([.01]*15 + [.03]*6 + [.002] + [.1]*8, dtype=np.float32)
    normalizer = Normalizer.fit(train_obs, floors)
    xval = torch.from_numpy(normalizer.transform(val_obs))
    yval = torch.from_numpy(val_actions)
    # Absolute position columns 0:9 are intentionally excluded for translation
    # generalization; no phase-selected teacher target/error is exposed.
    model = Policy(obs_size=train_obs.shape[1], hidden_size=hidden, feature_indices=range(9, 30))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    weights = torch.tensor([1., 1., 1., .25])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        noise_obs = np.stack([noisy_observation(o, rng, observation_noise) for o in train_obs]) if observation_noise else train_obs
        x = torch.from_numpy(normalizer.transform(noise_obs))
        y = torch.from_numpy(train_actions)
        order = torch.randperm(len(x))
        loss_sum = 0.
        for batch in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[batch])
            loss = ((predicted - y[batch]).square() * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            pred = model(xval)
            val_loss = float(((pred-yval).square()*weights).mean())
            cart_mae = float((pred[:, :3]-yval[:, :3]).abs().mean())
            grip_accuracy = float(((pred[:, 3] > 0) == (yval[:, 3] > 0)).float().mean())
        history.append({"epoch": epoch+1, "train_loss": loss_sum/len(x), "val_loss": val_loss,
                        "val_cartesian_normalized_mae": cart_mae, "val_gripper_sign_accuracy": grip_accuracy})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch+1
            best_state = copy.deepcopy(model.state_dict())
        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"epoch={epoch+1} train={loss_sum/len(x):.6f} val={val_loss:.6f} cart_mae={cart_mae:.5f} grip={grip_accuracy:.3f}", flush=True)
        if epoch+1-best_epoch >= 60:
            break
    model.load_state_dict(best_state)
    meta = {"seed": seed, "device": "cpu", "parameter_count": model.parameter_count,
            "training_samples": len(train_obs), "validation_samples": len(val_obs),
            "training_sources": sorted(train_sources), "validation_sources": sorted(val_sources),
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "observation_noise_std_m": observation_noise, "feature_indices": model.feature_indices,
            "architecture": f"21->{hidden}->{hidden}->4 tanh", "duration_s": time.monotonic()-start,
            "label_source": "synthetic_robot_teacher_actions_from_video_derived_scene_configs",
            "no_test_data_used_for_model_selection": True}
    save_checkpoint(output_dir / "policy.pt", model, normalizer, meta)
    write_json(output_dir / "training.json", {"metadata": meta, "history": history})
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return model, normalizer, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/dataset")
    parser.add_argument("--output", default="outputs/model")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--noise", type=float, default=.0002)
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.hidden, args.seed, args.noise)


if __name__ == "__main__":
    main()
