"""Quantify and plot the measured real paths and executed V3 transfer paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import write_json
from .integrity import sha256_file
from .video_policy import load_video_policy


def lateral_deviation(xy):
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return 0.
    direction = xy[-1]-xy[0]
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return 0.
    normal = np.array([-direction[1], direction[0]])/length
    return float(np.max(np.abs((xy-xy[0]) @ normal)))


def analyze(motion, checkpoint, rollouts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, meta = load_video_policy(checkpoint)
    payload = json.loads(Path(motion).read_text(encoding='utf-8'))
    if sha256_file(motion) != meta['motion_sha256']:
        raise ValueError('Analysis motion differs from frozen model supervision')
    rows = json.loads(Path(rollouts).read_text(encoding='utf-8'))
    path_stats = []
    for row in rows:
        path_stats.append(dict(configuration_id=row['configuration_id'], controller=row['controller'],
                               success=row['success'],
                               ring_max_lateral_deviation_m=lateral_deviation(row['transfer_ring_xy_m']),
                               command_max_lateral_deviation_m=lateral_deviation(row['transfer_command_xy_m'])))
    summary = {}
    for name in ('learned_motion', 'straight_path'):
        group = [r for r in path_stats if r['controller'] == name]
        summary[name] = dict(cases=len(group),
                            mean_ring_max_lateral_deviation_m=float(np.mean([r['ring_max_lateral_deviation_m'] for r in group])),
                            mean_command_max_lateral_deviation_m=float(np.mean([r['command_max_lateral_deviation_m'] for r in group])))
    report = dict(checkpoint_sha256=sha256_file(checkpoint), rollouts_sha256=sha256_file(rollouts),
                  summary=summary, per_case=path_stats,
                  interpretation='Deviation demonstrates that the trained model changes executed paths; '
                                 'it does not establish better task success or calibrated 3D accuracy.')
    write_json(output/'path_contribution.json', report)
    accepted = [r for r in payload['episodes'] if r['status'] == 'accepted']
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.subplots_adjust(top=.87, bottom=.07, left=.07, right=.98, hspace=.65, wspace=.35)
    for ax, record in zip(axes.flat, accepted):
        p = record['path']
        xy = np.asarray(p['relative_xy_m'])*1000
        progress = np.asarray(p['progress'])
        points = np.array([model.waypoint(xy[0]/1000, xy[-1]/1000, float(t)) for t in np.linspace(0, 1, 101)])*1000
        kinds = np.array(p['sample_kind'])
        ax.plot(xy[:, 0], xy[:, 1], color='#27614d', lw=1, label='Extracted path')
        ax.scatter(xy[kinds == 'observed', 0], xy[kinds == 'observed', 1], color='#27614d', s=16, label='Observed')
        ax.scatter(xy[kinds == 'interpolated', 0], xy[kinds == 'interpolated', 1], color='#c18528', marker='x', s=25, label='Gap interpolation')
        ax.plot(points[:, 0], points[:, 1], color='#386fc4', lw=1.6, label='Frozen model')
        ax.plot(xy[[0, -1], 0], xy[[0, -1], 1], color='#888888', ls='--', label='Straight baseline')
        split = next(k for k, ids in meta['source_splits'].items() if record['episode_id'] in ids)
        ax.set_title(f'{record["episode_id"]} / {split}')
        ax.set_xlabel('Approximate bearing-relative X (mm)')
        ax.set_ylabel('Approximate image-plane Y (mm)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=5, fontsize=9)
    fig.suptitle('Real video paths versus the frozen model: 5 train, 1 validation, 3 held-out', y=.99, fontsize=14)
    fig.savefig(output/'real_path_predictions.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    first_id = rows[0]['configuration_id']
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for name, color in (('learned_motion', '#386fc4'), ('straight_path', '#888888')):
        row = next(r for r in rows if r['configuration_id'] == first_id and r['controller'] == name)
        xy = np.asarray(row['transfer_ring_xy_m'])*1000
        ax.plot(xy[:, 0], xy[:, 1], label=name, color=color, lw=2)
    ax.set_title(f'Physical ring transfer: first frozen case {first_id}')
    ax.set_xlabel('Simulator X (mm)'); ax.set_ylabel('Simulator Y (mm)')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/'physical_transfer_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', default='data/real_motion.json')
    parser.add_argument('--checkpoint', default='outputs/video_model/policy.pt')
    parser.add_argument('--rollouts', default='outputs/video_v3/evaluation/rollouts.json')
    parser.add_argument('--output', default='outputs/video_v3/analysis')
    args = parser.parse_args()
    analyze(args.motion, args.checkpoint, args.rollouts, args.output)


if __name__ == '__main__':
    main()
