"""Replay six predeclared geometry cases, retaining failures in the video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np

from .dataset import write_json
from .evaluate import save_video
from .generalization_control import GeneralizationController, GeneralizationEnv
from .generalization_evaluate import validate_manifest
from .generalization_policy import load_generalization_policy
from .integrity import environment_signature, sha256_file
from .sim import CONTROL_DT


DEMO_INDICES = (0, 16, 32, 48, 64, 80)


def demo(manifest_path, checkpoint, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if manifest['environment'] != environment_signature():
        raise ValueError('Demo scene physics differs from frozen protocol')
    model, metadata = load_generalization_policy(checkpoint)
    if metadata['development_annotations_sha256'] != manifest['development_annotations_sha256']:
        raise ValueError('Demo model data differs from frozen scenes')
    cases = manifest['configurations']
    # The evaluator stores a single procedural list, not human descendants.
    if isinstance(cases, dict):
        cases = cases['test']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clips_directory = output.with_suffix('')
    clips_directory.mkdir(parents=True, exist_ok=True)
    env = GeneralizationEnv(render_width=480, render_height=360)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .37, 120, -33
    rows, clips, screenshots = [], [], []
    try:
        for index in DEMO_INDICES:
            cfg = cases[index]
            obs = env.reset(cfg['ring_xy_m'], cfg['bearing_xy_m'], cfg['reset_seed'])
            center = (np.asarray(cfg['ring_xy_m'])+cfg['bearing_xy_m'])/2
            camera.lookat[:] = [center[0], center[1], .145]
            controller, frames = GeneralizationController(model), []
            for step in range(env.max_steps):
                obs, done, info = env.step(controller(env, obs))
                renderer.update_scene(env.data, camera=camera)
                frame = np.concatenate([env.render(), renderer.render().copy()], axis=1)
                banner = np.full((54, frame.shape[1], 3), (25, 30, 37), dtype=np.uint8)
                frame = np.concatenate([banner, frame], axis=0)
                displacement = np.asarray(cfg['bearing_xy_m'])-cfg['ring_xy_m']
                label = (f'V4 | {cfg["configuration_id"]} | distance {np.linalg.norm(displacement)*1000:.1f} mm '
                         f'| direction {np.degrees(np.arctan2(displacement[1], displacement[0]))%360:.1f} deg')
                cv2.putText(frame, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
                cv2.putText(frame, f'{env.phase_name} | Learned XY; planned grasp/Z/fingers; free-body contact physics',
                            (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)
                frames.append(frame)
                if done:
                    break
            success = env.success()
            cv2.putText(frames[-1], 'SUCCESS' if success else 'FAILURE', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (90, 240, 130) if success else (255, 110, 100), 2)
            # Hold the terminal frame visibly, including failed attempts.
            frames.extend([frames[-1]]*10)
            clip = clips_directory/f'case_{index:02d}.mp4'
            save_video(clip, frames, round(1/CONTROL_DT))
            clips.append(clip.resolve())
            screenshots.append(frames[-1])
            rows.append(dict(index=index, configuration_id=cfg['configuration_id'], success=success,
                             steps=step+1, learned_reference_evaluations=controller.learned_reference_evaluations,
                             transfer_steps=controller.transfer_steps, lag_pause_steps=controller.lag_pause_steps, info=info))
            print(f'Demo case {index}: success={success}', flush=True)
        playlist = clips_directory/'concat.txt'
        playlist.write_text(''.join("file '"+str(path).replace('\\', '/').replace("'", "'\\''")+"'\n" for path in clips), encoding='utf-8')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(playlist), '-c', 'copy', str(output)], check=True)
        montage = np.concatenate([np.concatenate(screenshots[i:i+2], axis=1) for i in range(0, 6, 2)], axis=0)
        cv2.imwrite(str(output.with_suffix('.jpg')), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        result = dict(checkpoint_sha256=sha256_file(checkpoint), manifest_sha256=sha256_file(manifest_path),
                      cases=rows, selection='Fixed indices 0,16,32,48,64,80 before examining test outcomes; '
                      'deterministic replay of existing benchmark cases, no additional trial count',
                      controller_scope='Learned optical XY transfer; geometric grasp, Z and fingers; simulator state observations')
        write_json(output.with_suffix('.json'), result)
        return result
    finally:
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='outputs/generalization_v4/configurations.json')
    parser.add_argument('--checkpoint', default='outputs/generalization_v4/model/policy.json')
    parser.add_argument('--output', default='outputs/generalization_v4/student_demo.mp4')
    args = parser.parse_args()
    demo(args.manifest, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
