"""Preset TinyVLA physical replay: initial image/text, then proprioceptive execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from .dataset import write_json
from .evaluate import save_video
from .integrity import sha256_file
from .vla_control import VLAController
from .vla_evaluate import validate_manifest,validate_receipt
from .vla_env import VLAEnv,score_target
from .vla_policy import PRIMARY_INSTRUCTIONS,load_vla_policy


DEMO_CASES = ((0,'inside'),(32,'inside'),(64,'inside'),(0,'left'),(0,'right'),(48,'inside'))


def demo(manifest_path,checkpoint,output):
    torch.set_num_threads(2)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    receipt = json.loads(Path(manifest_path).with_name('model_freeze.json').read_text(encoding='utf-8'))
    validate_receipt(receipt,manifest)
    if receipt['checkpoint_sha256'] != sha256_file(checkpoint):
        raise ValueError('Demo model differs from frozen experiment')
    cases = manifest['configurations']
    model,_ = load_vla_policy(checkpoint)
    env = VLAEnv(render_width=480,render_height=360)
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    clips_dir = output.with_suffix(''); clips_dir.mkdir(parents=True,exist_ok=True)
    rows,clips,screens = [],[],[]
    try:
        for clip_index,(index,task) in enumerate(DEMO_CASES):
            cfg = cases[index]
            env.reset(cfg['ring_xy_m'],cfg['bearing_xy_m'],cfg['reset_seed'])
            initial = env.render_policy()
            controller = VLAController(model,PRIMARY_INSTRUCTIONS[task])
            frames = []
            for step in range(env.max_steps):
                _,done,info = env.step(controller(env))
                if step%2 == 0 or controller.finished or done:
                    left = np.full((360,360,3),235,np.uint8)
                    left[52:308,52:308] = initial
                    frame = np.concatenate([left,env.render()],axis=1)
                    banner = np.full((54,840,3),[25,30,37],np.uint8)
                    frame = np.concatenate([banner,frame],axis=0)
                    cv2.putText(frame,f'TinyVLA | {cfg["configuration_id"]} | {PRIMARY_INSTRUCTIONS[task]}',
                                (10,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
                    cv2.putText(frame,f'Initial RGB -> learned23-waypoint plan | Own-state servo | waypoint {min(controller.waypoint_index+1,23)}/23',
                                (10,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
                    frames.append(frame)
                if controller.finished or done:
                    break
            success = bool(controller.finished and score_target(env,PRIMARY_INSTRUCTIONS[task]))
            cv2.putText(frames[-1],'SUCCESS' if success else 'FAILURE',(375,87),
                        cv2.FONT_HERSHEY_SIMPLEX,.8,(80,235,120) if success else (255,100,90),2)
            frames.extend([frames[-1]]*10)
            clip = clips_dir/f'case_{clip_index}.mp4'
            save_video(clip,frames,10)
            clips.append(clip.resolve()); screens.append(frames[-1])
            rows.append(dict(index=index,task=task,success=success,steps=step+1,
                configuration_id=cfg['configuration_id'],neural_forward_calls=controller.neural_forward_calls,
                predicted_waypoints=controller.predicted_waypoints.tolist(),info=info))
            print(f'Demo {index}/{task}: success={success}',flush=True)
        playlist = clips_dir/'concat.txt'
        playlist.write_text(''.join("file '"+str(p).replace('\\','/')+"'\n" for p in clips),encoding='utf-8')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy',str(output)],check=True)
        montage = np.concatenate([np.concatenate(screens[i:i+2],axis=1) for i in (0,2,4)],axis=0)
        cv2.imwrite(str(output.with_suffix('.jpg')),cv2.cvtColor(montage,cv2.COLOR_RGB2BGR))
        result = dict(cases=rows,checkpoint_sha256=sha256_file(checkpoint),manifest_sha256=sha256_file(manifest_path),
            scope='Hierarchical finite-instruction VLA; initial robot-hidden RGB only; neuralXYZ/grip plan; hand-coded waypoint schedule/servo',
            selection='Preset indices/tasks; retain failures; these repeat already-scored scenes, not independent trials')
        write_json(output.with_suffix('.json'),result)
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='outputs/vla_v5/configurations.json')
    parser.add_argument('--checkpoint',default='outputs/vla_v5/model/policy.pt')
    parser.add_argument('--output',default='outputs/vla_v5/student_demo.mp4')
    args = parser.parse_args(); demo(args.manifest,args.checkpoint,args.output)
