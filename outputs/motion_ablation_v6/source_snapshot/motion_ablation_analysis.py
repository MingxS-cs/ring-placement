"""Compare saved paired rollouts, without feeding metrics back into control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .integrity import sha256_file

VARIANTS = ('video_motion', 'straight_transfer')


def transfer_path(row, field='ring_xyz_m'):
    stages = row.get('controller_waypoint_indices', [])
    trajectory = row.get('trajectory', [])
    indices = [i for i, stage in enumerate(stages[:max(0, len(trajectory)-1)]) if 4 <= stage <= 19]
    if not indices:
        return np.empty((0, 2))
    return np.asarray([point[field][:2] for point in trajectory[indices[0]:indices[-1]+2]], float)


def sample_by_arc_length(points, samples=100):
    points = np.asarray(points, float)
    if not len(points):
        return None
    progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if progress[-1] == 0:
        return np.tile(points[0], (samples, 1))
    keep = np.r_[True, np.diff(progress) > 0]
    return np.column_stack([np.interp(np.linspace(0, progress[-1], samples), progress[keep], points[keep, axis])
                            for axis in range(2)])


def path_stats(points):
    if len(points) < 2:
        return None
    vector = points[-1]-points[0]
    chord = float(np.linalg.norm(vector))
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    offset = points-points[0]
    lateral = None if chord <= 1e-9 else float(np.abs(vector[0]*offset[:, 1]-vector[1]*offset[:, 0]).max()/chord)
    return dict(path_length_mm=length*1000, max_lateral_mm=None if lateral is None else lateral*1000,
                arc_to_chord=None if chord <= 1e-9 else length/chord)


def audit_pairs(report, expected_pairs=96):
    rows = report['rollouts']
    pairs = {}
    unavailable_proof = 0
    for row in rows:
        if row.get('policy_forbidden_access_attempts', 0):
            raise ValueError('A controller accessed forbidden state')
        proof = (row.get('input_image_sha256') and row.get('predicted_waypoints') is not None
                 and row.get('neural_forward_calls') == 1 and row.get('steps', 0) > 0
                 and len(row.get('trajectory', [])) == row['steps']+1
                 and len(row.get('controller_waypoint_indices', [])) == row['steps'])
        if row.get('controller_finished') and not proof:
            raise ValueError('Finished execution lacks observation/action/trajectory evidence')
        if row.get('success') and (not row.get('controller_finished') or not row.get('target_score')):
            raise ValueError('Success lacks completed execution and target score')
        unavailable_proof += not bool(proof)
        group = pairs.setdefault(row['configuration_id'], {})
        if row['suite'] not in VARIANTS or row['suite'] in group:
            raise ValueError('Unregistered variant or repeated attempt')
        group[row['suite']] = row
    if len(rows) != 2*expected_pairs or len(pairs) != expected_pairs or any(set(p) != set(VARIANTS) for p in pairs.values()):
        raise ValueError('All registered pairs, including failures, are required')
    shared = np.ones((23, 4), bool); shared[4:19, :2] = False
    same_images = common = changed = 0
    rms, chunk_changes, path_metrics = [], [], {name: [] for name in VARIANTS}
    contingency = {'both_success': 0, 'video_only': 0, 'straight_only': 0, 'neither': 0}
    for pair in pairs.values():
        video, straight = (pair[name] for name in VARIANTS)
        if any(video.get(key) != straight.get(key) for key in ('ring_xy_m', 'bearing_xy_m', 'reset_seed', 'instruction')):
            raise ValueError('Matched reset geometry or instruction differs')
        images = [r.get('input_image_sha256') for r in (video, straight)]
        if all(images):
            if images[0] != images[1]:
                raise ValueError('Matched observations differ')
            same_images += 1
        chunks = [r.get('predicted_waypoints') for r in (video, straight)]
        if all(chunk is not None for chunk in chunks):
            a, b = (np.asarray(chunk, float) for chunk in chunks)
            if (a.shape != (23, 4) or b.shape != a.shape or not np.isfinite(a).all()
                    or not np.isfinite(b).all() or not np.array_equal(a[shared], b[shared])):
                raise ValueError('A common grasp, endpoint, Z or gripper action changed')
            common += 1
            difference = float(np.abs(a[~shared]-b[~shared]).max())*1000
            chunk_changes.append(difference); changed += difference > .001
        paths = [transfer_path(row) for row in (video, straight)]
        for name, path in zip(VARIANTS, paths):
            stats = path_stats(path)
            if stats is not None:
                path_metrics[name].append(stats)
        samples = [sample_by_arc_length(path) for path in paths]
        if all(sample is not None for sample in samples):
            rms.append(float(np.sqrt(np.square(samples[0]-samples[1]).sum(axis=1).mean()))*1000)
        flags = (bool(video['success']), bool(straight['success']))
        key = {(True, True): 'both_success', (True, False): 'video_only',
               (False, True): 'straight_only', (False, False): 'neither'}[flags]
        contingency[key] += 1
    def mean(values):
        return float(np.mean(values)) if values else None
    metrics = {name: dict(available_paths=len(items), total_attempts=expected_pairs,
                         **{key+'_mean': mean([item[key] for item in items if item[key] is not None])
                            for key in ('path_length_mm', 'max_lateral_mm', 'arc_to_chord')})
               for name, items in path_metrics.items()}
    return dict(status='incomplete' if unavailable_proof else 'passed', evidence_unavailable_attempts=unavailable_proof,
                attempts=len(rows), pairs=len(pairs), same_images=same_images,
                common_actions_equal=common, changed_transfer_pairs=changed,
                successes={name: sum(bool(row['success']) for row in rows if row['suite'] == name) for name in VARIANTS},
                paired_success=contingency, forbidden_accesses=sum(row.get('policy_forbidden_access_attempts', 0) for row in rows),
                maximum_chunk_change_mm=max(chunk_changes) if chunk_changes else None,
                mean_chunk_change_mm=mean(chunk_changes), paired_actual_path_rms_mm_mean=mean(rms),
                available_path_rms_pairs=len(rms), actual_ring_transfer=metrics,
                interpretation='Partial curvature intervention; endpoint geometry still comes from video. Paths use executor stages, including failed attempts.')


def analyze(report_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new analysis directory to preserve saved evidence')
    import matplotlib.pyplot as plt
    from .motion_ablation import summarize, validate_freeze_receipt

    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    manifest_path = Path(report_path).parent/'manifest.json'
    receipt_path = manifest_path.with_name('freeze_receipt.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    validate_freeze_receipt(receipt, manifest_path)
    expected_models = {name: manifest['models'][name]['sha256'] for name in VARIANTS}
    if (report.get('manifest_sha256') != sha256_file(manifest_path)
            or report.get('freeze_receipt_sha256') != sha256_file(receipt_path)
            or report.get('model_sha256') != expected_models
            or report.get('summary') != summarize(report['rollouts'], manifest['schedule'])):
        raise ValueError('Saved evaluation differs from the frozen experiment or its registered schedule')
    result = audit_pairs(report)
    output.mkdir(parents=True, exist_ok=False)
    rows = report['rollouts']
    identifiers = list(dict.fromkeys(row['configuration_id'] for row in rows))
    lookup = {(row['configuration_id'], row['suite']): row for row in rows}
    colors = {'video_motion': '#286b59', 'straight_transfer': '#c06d2a'}
    labels = {'video_motion': 'Video-derived curve', 'straight_transfer': 'Straight transfer'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, index in zip(axes, (0, 32, 64)):
        for name in VARIANTS:
            row = lookup[identifiers[index], name]
            path = transfer_path(row)
            if len(path):
                axis.plot(path[:, 0]*1000, path[:, 1]*1000, color=colors[name], label=labels[name])
                axis.scatter(path[[0, -1], 0]*1000, path[[0, -1], 1]*1000, color=colors[name], s=12)
        axis.set(title=f'Preset scene {index}', xlabel='Ring X (mm)', ylabel='Ring Y (mm)')
        axis.set_aspect('equal', adjustable='datalim'); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual ring transfer paths: same RGB, grasp, height and executor')
    fig.savefig(output/'paired_ring_paths.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for name in VARIANTS:
        selected = [row for row in rows if row['suite'] == name]
        stats = [path_stats(transfer_path(row)) for row in selected]
        valid = [(row, item) for row, item in zip(selected, stats) if item and item['max_lateral_mm'] is not None]
        axes[0].scatter([np.linalg.norm(np.asarray(row['bearing_xy_m'])-row['ring_xy_m'])*1000 for row, _ in valid],
                        [item['max_lateral_mm'] for _, item in valid], color=colors[name], s=14, alpha=.7, label=labels[name])
    axes[0].set(xlabel='Initial ring-to-bearing distance (mm)', ylabel='Actual maximum lateral departure (mm)')
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    axes[1].bar([labels[name] for name in VARIANTS], [result['successes'][name] for name in VARIANTS],
                color=[colors[name] for name in VARIANTS])
    axes[1].set(ylim=(0, 105), ylabel='Successes / 96 registered attempts')
    for index, name in enumerate(VARIANTS):
        axes[1].text(index, result['successes'][name]+2, f"{result['successes'][name]}/96", ha='center')
    fig.suptitle('Path differences and task outcomes are separate measurements')
    fig.savefig(output/'motion_and_success.png', dpi=160); plt.close(fig)
    result['evaluation_sha256'] = sha256_file(report_path)
    (output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='outputs/motion_ablation_v6/evaluation.json')
    parser.add_argument('--output', default='outputs/motion_ablation_v6/analysis')
    args = parser.parse_args()
    analyze(args.report, args.output)
