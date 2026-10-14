"""Replay the first frozen student test case with overview and close views."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .dataset import write_json
from .evaluate import save_video
from .integrity import verify_binding, sha256_file
from .policy import load_checkpoint
from .sim import CONTROL_DT, RingEnv


def demo(dataset, checkpoint, output):
    model, norm, meta = load_checkpoint(checkpoint)
    verify_binding(meta["binding"], Path(dataset)/"configurations.json")
    manifest = json.loads((Path(dataset)/"configurations.json").read_text(encoding="utf-8"))
    cfg = manifest["configurations"]["test"][0]
    env = RingEnv()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    center = (np.array(cfg["ring_xy_m"])+np.array(cfg["bearing_xy_m"]))/2
    cam.lookat[:] = [center[0], center[1], .15]
    cam.distance = .33
    cam.azimuth = 120
    cam.elevation = -30
    frames = []
    phase_frames = {}
    try:
        obs = env.reset(cfg["ring_xy_m"], cfg["bearing_xy_m"], cfg["reset_seed"])
        for step in range(env.max_steps):
            phase = env.phase_name
            action = model.act(obs, norm)
            obs, done, info = env.step(action)
            wide = env.render()
            renderer.update_scene(env.data, camera=cam)
            close = renderer.render().copy()
            frame = np.concatenate([wide, close], axis=1)
            cv2.putText(frame, f"Student | {cfg['configuration_id']} | {phase}",
                        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            frames.append(frame)
            if phase in ("grasp", "lift", "transfer", "release", "retract"):
                phase_frames[phase] = close.copy()
            if done:
                break
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_video(output, frames, round(1/CONTROL_DT))
        for phase, frame in phase_frames.items():
            cv2.imwrite(str(output.with_name(f"student_{phase}.jpg")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        result = {"configuration_id": cfg["configuration_id"], "source_episode_id": cfg["source_episode_id"],
                  "checkpoint_sha256": sha256_file(checkpoint), "success": env.success(), "steps": step+1,
                  "teacher_actions": 0, "purpose": "Deterministic replay of first final-test case; no additional benchmark cases", "info": info}
        write_json(output.with_suffix(".json"), result)
        print(json.dumps(result))
    finally:
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/dataset_dagger")
    parser.add_argument("--checkpoint", default="outputs/model_dagger/policy.pt")
    parser.add_argument("--output", default="outputs/evaluation_final/student_closeup.mp4")
    args = parser.parse_args()
    demo(args.dataset, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
