"""Plot frozen coverage, physical motion and unchanged held-out fidelity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import write_json
from .integrity import sha256_file
from .video_analysis import lateral_deviation


COLORS = dict(v4='#24715a', straight='#777777', v3_profile='#4474bc', v3='#4474bc')


def analyze(manifest_path, rollouts_path, fidelity_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, rows, fidelity = [json.loads(Path(path).read_text(encoding='utf-8'))
                                for path in (manifest_path, rollouts_path, fidelity_path)]
    if fidelity['provenance']['selected_checkpoint_sha256'] != sha256_file('outputs/generalization_v4/model/policy.json'):
        raise ValueError('Plot fidelity does not belong to delivered frozen model')
    configs = manifest['configurations']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for band, ax in enumerate(axes):
        group = [c for c in configs if c['distance_band_index'] == band]
        ring, bearing = np.array([c['ring_xy_m'] for c in group])*1000, np.array([c['bearing_xy_m'] for c in group])*1000
        ax.scatter(ring[:, 0], ring[:, 1], marker='o', s=22, color=COLORS['v4'], label='Ring starts')
        ax.scatter(bearing[:, 0], bearing[:, 1], marker='x', s=23, color=COLORS['v3'], label='Bearing targets')
        for start, end in zip(ring, bearing):
            ax.plot([start[0], end[0]], [start[1], end[1]], color='#aaaaaa', alpha=.25, lw=.6)
        lower, upper = manifest['geometry_protocol']['distance_bands_m'][band]
        ax.set_title(f'{lower*1000:.0f}-{upper*1000:.0f} mm | 32 scenes / 8 directions')
        ax.set(xlabel='Simulator X (mm)', ylabel='Simulator Y (mm)', xlim=(315,625), ylim=(-235,235))
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('96 frozen procedural scenes: same objects and fixed reference')
    fig.savefig(output/'configuration_coverage.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, row in zip(axes, fidelity['per_source']):
        xy = np.array(row['observed_relative_xy_m'])*1000
        ax.plot(xy[:, 0], xy[:, 1], 'o-', color='#202e3b', lw=1, ms=4, label='Observed optical path')
        for name in ('v4', 'straight', 'v3'):
            prediction = np.array(row['predicted_relative_xy_m'][name])*1000
            ax.plot(prediction[:, 0], prediction[:, 1], color=COLORS[name], lw=1.7, ls='--' if name=='straight' else '-', label=name)
        ax.set_title(f'{row["source_episode_id"]} | {row["intermediate_points"]} intermediate observations')
        ax.set(xlabel='Bearing-relative optical X (mm)', ylabel='Approximate optical Y (mm)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Locked reserved paths: V4 improves over V3, remains worse than straight on mean error')
    fig.savefig(output/'holdout_path_predictions.png', dpi=155)
    plt.close(fig)

    selected = (0,16,32,48,64,80)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    for ax, index in zip(axes.flat, selected):
        cfg = configs[index]
        for name in ('v4', 'straight', 'v3_profile'):
            row = next(r for r in rows if r['configuration_id']==cfg['configuration_id'] and r['controller']==name)
            xy = np.array(row['transfer_ring_xy_m'])*1000
            ax.plot(xy[:,0], xy[:,1], lw=1.5, color=COLORS[name], label=name)
        ax.set_title(cfg['configuration_id']+f' | {cfg["endpoint_distance_m"]*1000:.1f} mm')
        ax.set(xlabel='Physical ring X (mm)', ylabel='Physical ring Y (mm)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Actual free-body ring paths: preset cases shared with the demonstration video')
    fig.savefig(output/'physical_transfer_comparison.png', dpi=155)
    plt.close(fig)
    contributions = {}
    for name in ('v4', 'straight', 'v3_profile'):
        group = [r for r in rows if r['controller']==name]
        contributions[name] = dict(cases=len(group), successes=sum(r['success'] for r in group),
            mean_ring_max_lateral_deviation_m=float(np.mean([lateral_deviation(r['transfer_ring_xy_m']) for r in group])),
            mean_command_max_lateral_deviation_m=float(np.mean([lateral_deviation(r['transfer_command_xy_m']) for r in group])))
    report = dict(manifest_sha256=sha256_file(manifest_path), rollouts_sha256=sha256_file(rollouts_path),
                  fidelity_sha256=sha256_file(fidelity_path), physical_path_contribution=contributions,
                  interpretation='Physical path difference demonstrates data-to-motion effect; equal task success '
                  'does not establish learned placement advantage. Optical paths are approximate, not calibrated 3D.')
    write_json(output/'analysis.json', report)
    print(json.dumps(contributions, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='outputs/generalization_v4/configurations.json')
    parser.add_argument('--rollouts', default='outputs/generalization_v4/evaluation/rollouts.json')
    parser.add_argument('--fidelity', default='outputs/generalization_v4/fidelity/fidelity.json')
    parser.add_argument('--output', default='outputs/generalization_v4/analysis')
    args = parser.parse_args()
    analyze(args.manifest,args.rollouts,args.fidelity,args.output)


if __name__ == '__main__':
    main()
