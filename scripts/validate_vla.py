"""Development-only physical checks; never opens the fresh procedural test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from ring_learning.vla_control import VLAController
from ring_learning.vla_env import VLAEnv,score_target
from ring_learning.vla_policy import PRIMARY_INSTRUCTIONS,load_vla_policy
from ring_learning.vla_evaluate import ControllerEnvironmentView


def validate(checkpoint,output):
    torch.set_num_threads(2)
    model,metadata = load_vla_policy(checkpoint)
    manifest = json.loads(Path('outputs/vla_v5/dataset/manifest.json').read_text(encoding='utf-8'))
    env,rows = VLAEnv(),[]
    try:
        for task,instruction in PRIMARY_INSTRUCTIONS.items():
            for cfg in manifest['configurations']['val']:
                env.reset(cfg['ring_xy_m'],cfg['bearing_xy_m'],cfg['reset_seed'])
                controller = VLAController(model,instruction)
                view = ControllerEnvironmentView(env)
                error = None
                try:
                    for step in range(env.max_steps):
                        _,done,_ = env.step(controller(view))
                        if done or controller.finished:
                            break
                except Exception as exception:
                    error = type(exception).__name__+': '+str(exception)
                    step = env.steps
                success = error is None and controller.finished and score_target(env,instruction)
                rows.append(dict(task=task,configuration_id=cfg['configuration_id'],success=bool(success),
                    steps=step+1,exception=error,waypoint_index=controller.waypoint_index,
                    ring_xyz=env.ring_xyz.tolist(),predicted_waypoints=None if controller.predicted_waypoints is None else controller.predicted_waypoints.tolist()))
            print(task,sum(r['success'] for r in rows if r['task']==task),'/24',flush=True)
    finally:
        env.close()
    result = dict(checkpoint=str(checkpoint),validation_only=True,rows=rows,
        summary={task:dict(successes=sum(r['success'] for r in rows if r['task']==task),attempts=24) for task in PRIMARY_INSTRUCTIONS})
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='outputs/vla_v5/model/policy.pt')
    parser.add_argument('--output',default='outputs/vla_v5/model/physical_validation.json')
    args=parser.parse_args(); validate(args.checkpoint,args.output)
