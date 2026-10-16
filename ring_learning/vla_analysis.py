"""Plot saved spatial and language rollouts and check their provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .dataset import write_json
from .integrity import sha256_file
from .vla_evaluate import validate_manifest, validate_receipt


def analyze(base='outputs/vla_v5'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = Path(base)
    manifest, receipt, report = [
        json.loads((base/name).read_text(encoding='utf-8'))
        for name in ('configurations.json', 'model_freeze.json', 'evaluation.json')
    ]
    validate_receipt(receipt, manifest)
    if (report['checkpoint_sha256'] != receipt['checkpoint_sha256']
            or report['manifest_sha256'] != sha256_file(base/'configurations.json')):
        raise ValueError('Analysis inputs do not belong to frozen experiment')
    rollouts = report['rollouts']
    analysis_directory = base/'analysis'
    analysis_directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, distance_band in zip(axes, range(3)):
        success_counts = np.zeros((1, 8), int)
        for direction in range(8):
            success_counts[0, direction] = sum(
                rollout['success'] for rollout in rollouts
                if rollout['suite'] == 'main'
                and rollout['distance_band_index'] == distance_band
                and rollout['direction_bin_index'] == direction
            )
        ax.imshow(success_counts, vmin=0, vmax=4, cmap='Greens', aspect=1.3)
        for direction_index in range(8):
            ax.text(direction_index, 0, f'{success_counts[0, direction_index]}/4',
                    ha='center', va='center', color='white')
        ax.set_yticks([])
        ax.set_xticks(range(8))
        ax.set_xlabel('Direction bin')
        ax.set_title(('45-75 mm', '75-110 mm', '110-160 mm')[distance_band])
    fig.suptitle('Frozen TinyVLA primary test: 96/96; novel combinations under a fixed camera')
    fig.savefig(analysis_directory/'spatial_generalization.png', dpi=160)
    plt.close(fig)
    scene = manifest['configurations'][0]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for instruction_index, color in ((0, '#276b51'), (1, '#4776bf'), (2, '#b77035')):
        selected_rollout = next(
            rollout for rollout in rollouts
            if rollout['suite'] == 'language_counterfactual'
            and rollout['configuration_id'] == scene['configuration_id']
            and rollout['instruction_index'] == instruction_index
        )
        ring_xy_mm = np.array([state['ring_xyz_m'][:2] for state in selected_rollout['trajectory']])*1000
        ax.plot(ring_xy_mm[:, 0], ring_xy_mm[:, 1], color=color,
                label=('inside', 'left', 'right')[instruction_index], lw=2)
        ax.scatter(*ring_xy_mm[-1], color=color, s=40)
    ax.set(xlabel='Actual ring X (mm)', ylabel='Actual ring Y (mm)',
           title='Same initial RGB, different text: actual physical ring paths')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(analysis_directory/'language_physical_paths.png', dpi=160)
    plt.close(fig)
    center_errors = {}
    height_checks = {}
    for suite in ('main', 'black_image', 'language_counterfactual'):
        suite_rollouts = [rollout for rollout in rollouts if rollout['suite'] == suite]
        errors_m = []
        for rollout in suite_rollouts:
            final_state = rollout['trajectory'][-1]
            target_xy_m = np.array(final_state['bearing_xyz_m'][:2])+[
                (-.06 if rollout['instruction_index'] == 1
                 else .06 if rollout['instruction_index'] == 2 else 0.), 0.
            ]
            errors_m.append(np.linalg.norm(np.array(final_state['ring_xyz_m'][:2])-target_xy_m))
        center_errors[suite] = dict(mean_m=float(np.mean(errors_m)), max_m=float(np.max(errors_m)))
        height_checks[suite] = dict(minimum_case_max_height_m=min(
            max(state['ring_xyz_m'][2] for state in rollout['trajectory'])
            for rollout in suite_rollouts
        ))
    verification = dict(
        status='passed', checkpoint_sha256=receipt['checkpoint_sha256'],
        evaluation_sha256=sha256_file(base/'evaluation.json'),
        attempts=len(rollouts), all_completed=all(rollout['controller_finished'] for rollout in rollouts),
        forbidden_accesses=sum(rollout['policy_forbidden_access_attempts'] for rollout in rollouts),
        final_center_errors=center_errors, height_checks=height_checks, summary=report['summary']['by_suite'],
        paired_diagnostics=report['paired_diagnostics'],
        scope='Fixed robot-hidden simulator RGB, finite language, same objects; waypoint execution manual; no real-camera claims'
    )
    if len(rollouts) != 264 or verification['forbidden_accesses'] or not verification['all_completed']:
        raise ValueError('Incomplete or invalid saved experiment')
    write_json(analysis_directory/'verification.json', verification)
    print(json.dumps({key: value for key, value in verification.items()
                      if key not in ('paired_diagnostics', 'height_checks')}, indent=2))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='outputs/vla_v5')
    analyze(parser.parse_args().base)
