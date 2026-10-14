"""Deterministic first V3 test replay, with source video for visual comparison only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .dataset import write_json
from .evaluate import save_video
from .integrity import sha256_file
from .sim import CONTROL_DT, RingEnv
from .video_evaluate import validate_model_provenance
from .video_policy import VideoController, load_video_policy


def demo(manifest_path, checkpoint, output, source='data/source/359977334.mp4',
         motion='data/real_motion.json'):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    model, metadata = load_video_policy(checkpoint)
    validate_model_provenance(metadata, manifest, ['test'])
    cfg = manifest['configurations']['test'][0]
    if sha256_file(source) != manifest['source_video_sha256']:
        raise ValueError('Demo source video differs from original data')
    # These measured frames affect the left display panel only. The controller
    # below receives model + simulator state, and never receives this record.
    records = json.loads(Path(motion).read_text(encoding='utf-8'))['episodes']
    record = next(r for r in records if r['episode_id'] == cfg['source_episode_id'])
    path = record.get('path')
    env = RingEnv()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    center = (np.asarray(cfg['ring_xy_m'])+cfg['bearing_xy_m'])/2
    camera.lookat[:] = [center[0], center[1], .15]
    camera.distance, camera.azimuth, camera.elevation = .33, 120, -30
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open the bundled source video')
    frames, phase_frames = [], {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        obs = env.reset(cfg['ring_xy_m'], cfg['bearing_xy_m'], cfg['reset_seed'])
        controller = VideoController(model)
        for step in range(env.max_steps):
            old_phase = env.phase
            action = controller(env, obs)
            obs, done, info = env.step(action)
            if old_phase < 3 or path is None:
                source_frame = record['source_frame_start']
            elif old_phase == 3:
                source_frame = path['sample_source_frames'][0]
            elif old_phase == 4:
                source_frame = int(round(np.interp(controller.progress, path['progress'],
                                                  path['sample_source_frames'])))
            else:
                source_frame = record['source_frame_terminal']
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError('Cannot read the comparison frame')
            left = cv2.cvtColor(cv2.resize(image, (270, 480)), cv2.COLOR_BGR2RGB)
            wide = env.render()
            renderer.update_scene(env.data, camera=camera)
            close = renderer.render().copy()
            frame = np.concatenate([left, wide, close], axis=1)
            banner = np.full((50, frame.shape[1], 3), (25, 30, 37), dtype=np.uint8)
            frame = np.concatenate([banner, frame], axis=0)
            cv2.putText(frame, f'Real source {cfg["source_episode_id"]}', (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            cv2.putText(frame, 'Visual comparison only', (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1)
            cv2.putText(frame, f'Video-trained XY transfer | {cfg["configuration_id"]} | {env.phase_name}',
                        (285, 20), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
            cv2.putText(frame, 'Planned grasp/Z/gripper; free-body contact physics', (285, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1)
            frames.append(frame)
            if old_phase in (2, 3, 4, 6, 7):
                phase_frames[env.phase_name] = frame.copy()
            if done:
                break
        save_video(output, frames, round(1/CONTROL_DT))
        for phase, frame in phase_frames.items():
            cv2.imwrite(str(output.with_name(f'video_{phase}.jpg')), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        result = dict(configuration_id=cfg['configuration_id'], source_episode_id=cfg['source_episode_id'],
                      checkpoint_sha256=sha256_file(checkpoint), manifest_sha256=sha256_file(manifest_path),
                      success=env.success(), steps=step+1, learned_transfer_scope_steps=controller.learned_steps,
                      planned_stage_steps=controller.planned_steps, info=info,
                      source_panel_controls_robot=False,
                      selection='First frozen test case; deterministic replay, not an additional benchmark case',
                      scope='Learned XY transfer; planned grasp/lift/lower/release/Z/gripper')
        write_json(output.with_suffix('.json'), result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        cap.release()
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='outputs/video_v3/configurations.json')
    parser.add_argument('--checkpoint', default='outputs/video_model/policy.pt')
    parser.add_argument('--output', default='outputs/video_v3/student_demo.mp4')
    args = parser.parse_args()
    demo(args.manifest, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
