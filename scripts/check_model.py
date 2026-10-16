"""Run the three placement commands on validation scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ring_learning.vla_control import VLAController
from ring_learning.vla_env import VLAEnv, score_target
from ring_learning.vla_policy import PRIMARY_INSTRUCTIONS, load_vla_policy
from ring_learning.vla_evaluate import ControllerEnvironmentView


def validate(checkpoint, output):
    torch.set_num_threads(2)
    model, _ = load_vla_policy(checkpoint)
    dataset_manifest = json.loads(Path('outputs/vla_v5/dataset/manifest.json').read_text(encoding='utf-8'))
    env, validation_results = VLAEnv(), []
    try:
        for task, instruction in PRIMARY_INSTRUCTIONS.items():
            for scene in dataset_manifest['configurations']['val']:
                env.reset(scene['ring_xy_m'], scene['bearing_xy_m'], scene['reset_seed'])
                controller = VLAController(model, instruction)
                policy_view = ControllerEnvironmentView(env)
                failure_message = None
                try:
                    for step in range(env.max_steps):
                        _, done, _ = env.step(controller(policy_view))
                        if done or controller.finished:
                            break
                except Exception as simulation_error:
                    failure_message = type(simulation_error).__name__ + ': ' + str(simulation_error)
                    step = env.steps
                success = failure_message is None and controller.finished and score_target(env, instruction)
                validation_results.append(dict(
                    task=task,
                    configuration_id=scene['configuration_id'],
                    success=bool(success),
                    steps=step + 1,
                    exception=failure_message,
                    waypoint_index=controller.waypoint_index,
                    ring_xyz=env.ring_xyz.tolist(),
                    predicted_waypoints=(None if controller.predicted_waypoints is None
                                         else controller.predicted_waypoints.tolist()),
                ))
            print(task, sum(case_result['success'] for case_result in validation_results
                            if case_result['task'] == task), '/24', flush=True)
    finally:
        env.close()
    result = dict(
        checkpoint=str(checkpoint),
        validation_only=True,
        rows=validation_results,
        summary={task: dict(
            successes=sum(case_result['success'] for case_result in validation_results
                          if case_result['task'] == task),
            attempts=24,
        ) for task in PRIMARY_INSTRUCTIONS},
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/vla_v5/model/policy.pt')
    parser.add_argument('--output', default='outputs/vla_v5/model/physical_validation.json')
    args = parser.parse_args()
    validate(args.checkpoint, args.output)
