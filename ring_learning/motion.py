"""Recover visible image-plane ring motion with explicit gaps and source lineage.

This is approximate XY optical motion, not a recovered 3D human demonstration.
Known bearing diameter gives a local scale. Robot heights/orientation are supplied
by simulation. Shadow candidates are rejected with a temporal prediction gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .extract import DIMENSIONS_M, MAPPING_DISCLAIMER, _ellipse_candidates


def image_relative_xy(points_px, bearing_center_px, bearing_diameter_px):
    points = np.asarray(points_px, dtype=float)
    center = np.asarray(bearing_center_px, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or center.shape != (2,)
            or not np.isfinite(points).all() or not np.isfinite(center).all()
            or not np.isfinite(bearing_diameter_px) or bearing_diameter_px <= 0):
        raise ValueError('Finite XY samples and a positive bearing diameter are required')
    return (points - center) * [1, -1] * (.05 / bearing_diameter_px)


def build_path(relative_xy_m, source_frames):
    xy = np.asarray(relative_xy_m, dtype=float)
    frames = np.asarray(source_frames, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or frames.shape != (len(xy),)
            or not np.isfinite(xy).all() or not np.isfinite(frames).all()
            or np.any(np.diff(frames) <= 0) or np.any(frames != np.floor(frames))):
        raise ValueError('Finite XY points require strictly increasing integer source frames')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if arc[-1] <= 1e-8:
        raise ValueError('Stationary tracks cannot supply a transfer path')
    # Repeated optical positions remain in raw samples, but path interpolation
    # needs unique progress values. Preserve the final endpoint even on a plateau.
    keep = np.r_[np.diff(arc) > 1e-9, True]
    xy, frames, arc = xy[keep], frames[keep], arc[keep]
    progress = arc / arc[-1]
    line = xy[0] + progress[:, None] * (xy[-1] - xy[0])
    residual = xy - line
    return {'progress': progress.tolist(), 'relative_xy_m': xy.tolist(),
            'residual_xy_m': residual.tolist(), 'sample_source_frames': frames.astype(int).tolist(),
            'arc_length_m': float(arc[-1]),
            'straight_distance_m': float(np.linalg.norm(xy[-1] - xy[0])),
            'maximum_residual_m': float(np.linalg.norm(residual, axis=1).max()),
            'progress_definition': 'normalized cumulative image-plane XY arc length; not source time'}


def fill_short_gaps(samples, max_gap_frames=10):
    result = copy.deepcopy(samples)
    frames = np.asarray([item['source_frame'] for item in result], dtype=float)
    if len(frames) and (not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0)):
        raise ValueError('One episode with increasing source frames is required')
    observed = [i for i, item in enumerate(result) if item.get('center_px') is not None]
    for left, right in zip(observed, observed[1:]):
        if right == left + 1 or frames[right] - frames[left] > max_gap_frames:
            continue
        a, b = np.asarray(result[left]['center_px']), np.asarray(result[right]['center_px'])
        for index in range(left + 1, right):
            t = (frames[index] - frames[left]) / (frames[right] - frames[left])
            result[index].update(center_px=((1-t)*a+t*b).tolist(),
                                 kind='short_gap_interpolated', confidence=0.,
                                 interpolation_bracket_frames=[int(frames[left]), int(frames[right])],
                                 method='linear interpolation between measured centers within this episode')
    return result


def apply_sample_review(samples, review):
    result = copy.deepcopy(samples)
    rejected = set(review.get('reject_source_frames', []))
    if not rejected.issubset({s['source_frame'] for s in result}):
        raise ValueError('Reviewed rejection frame is not a sampled source frame')
    for item in result:
        if item['kind']=='short_gap_interpolated':
            item.update(center_px=None, kind='missing', confidence=0.)
            item.pop('interpolation_bracket_frames', None)
        if item['source_frame'] in rejected:
            item['rejected_center_px'] = item['center_px']
            item.update(center_px=None, kind='missing', confidence=0.,
                        manual_review_action='reject optical contour association')
    return fill_short_gaps(result, max_gap_frames=10)


def load_motion(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported motion schema')
    sha = payload.get('source', {}).get('sha256', '')
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('A source SHA-256 is required')
    records = {}
    for record in payload.get('episodes', []):
        identifier = record['episode_id']
        if identifier in records:
            raise ValueError('Duplicate episode source lineage')
        frames = np.asarray([s['source_frame'] for s in record['samples']], dtype=int)
        first, last = record['source_frame_start'], record['source_frame_terminal']
        if np.any(frames < first) or np.any(frames > last):
            raise ValueError('Motion sample crosses episode boundary')
        if len(frames) < 2 or np.any(np.diff(frames) <= 0):
            raise ValueError('Motion source frames must increase within one episode')
        if record.get('status') == 'accepted':
            p = record['path']
            checked = build_path(p['relative_xy_m'], p['sample_source_frames'])
            if (not np.allclose(checked['progress'], p['progress'])
                    or not np.allclose(checked['residual_xy_m'], p['residual_xy_m'])):
                raise ValueError('Motion path normalization is inconsistent')
            if not set(p['sample_source_frames']).issubset(set(frames.tolist())):
                raise ValueError('Path source frames lack sample lineage')
        records[identifier] = record
    return records


def detect_moving_ring(image, predicted_center_px, bearing_diameter_px, gate_px):
    """Closed dark contours near a temporal prediction, allowing visible inner holes.

    Low gray thresholds favor the solid ring over its much brighter cast shadow.
    A temporal gate rejects displaced annular shadows; missing evidence stays missing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = np.asarray(predicted_center_px, dtype=float)
    radius = int(gate_px + .75 * bearing_diameter_px)
    x0, y0 = np.maximum(np.floor(center-radius).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(center+radius).astype(int), [image.shape[1], image.shape[0]])
    roi = gray[y0:y1, x0:x1]
    if not roi.size:
        return None
    options = []
    for threshold in (55, 65, 80, 95, 110, 125):
        mask = cv2.inRange(roi, 0, threshold)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        for item in _ellipse_candidates(mask, .24*bearing_diameter_px,
                                        .67*bearing_diameter_px, .55):
            item['center_px'] = (np.asarray(item['center_px']) + [x0, y0]).tolist()
            distance = np.linalg.norm(np.asarray(item['center_px']) - center)
            if distance > gate_px:
                continue
            # Dark-threshold contour plus temporal agreement; confidence is heuristic.
            item['threshold_gray'] = threshold
            item['confidence'] = float(min(.90, item['circularity']) * (1-distance/(2*gate_px)))
            item['cost'] = float(distance + .075 * (threshold-55))
            item['method'] = 'visible closed dark ellipse with temporal gate; outer wall or inner aperture'
            options.append(item)
    best = min(options, key=lambda item: item['cost'], default=None)
    if best:
        best.pop('cost')
    return best


def _track_episode(cap, episode, fps, stride):
    first, last = episode['source_frame_start'], episode['source_frame_terminal']
    frame_indices = list(range(first, last+1, stride))
    if frame_indices[-1] != last:
        frame_indices.append(last)
    diameter = episode['bearing_outer_diameter_px']
    samples, history = [], []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Cannot read source frame {frame_index}')
        item = {'source_frame': frame_index,
                'source_pts_s': float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000,
                'episode_time_s': (frame_index-first)/fps,
                'center_px': None, 'kind': 'missing', 'confidence': 0.}
        if frame_index in (first, last):
            endpoint = 'initial' if frame_index == first else 'terminal'
            detection = copy.deepcopy(episode['detection'][endpoint]['ring'])
            detection['method'] = 'previously visually reviewed visible endpoint: ' + detection['method']
        else:
            prev_frame, prev_center = history[-1]
            predicted = prev_center.copy()
            if len(history) > 1:
                older_frame, older_center = history[-2]
                velocity = (prev_center-older_center)/(prev_frame-older_frame)
                predicted += velocity * min(frame_index-prev_frame, 8)
            gap = frame_index-prev_frame
            detection = detect_moving_ring(image, predicted, diameter,
                                           .20*diameter + 12*min(gap, 8))
        if detection:
            item.update(center_px=detection['center_px'], kind='observed',
                        confidence=detection['confidence'], evidence=detection)
            history.append((frame_index, np.asarray(detection['center_px'])))
        samples.append(item)
    return fill_short_gaps(samples, max_gap_frames=10)


def _make_record(episode, samples):
    record = {key: episode[key] for key in ('episode_id', 'scene_group', 'source_frame_start',
              'source_frame_terminal', 'bearing_center_px', 'bearing_outer_diameter_px')}
    record.update(samples=samples, status='rejected', rejection_reason=None)
    present = [i for i, item in enumerate(samples) if item['center_px'] is not None]
    xy = image_relative_xy([samples[i]['center_px'] for i in present],
                           episode['bearing_center_px'], episode['bearing_outer_diameter_px'])
    for index, point in zip(present, xy):
        samples[index]['relative_xy_m'] = point.tolist()
    # Retain the movement interval instead of including seconds of idle jitter.
    distance_start = np.linalg.norm(xy - xy[0], axis=1)
    distance_end = np.linalg.norm(xy - xy[-1], axis=1)
    active = np.flatnonzero((distance_start > .005) & (distance_end > .005))
    if not len(active):
        record['rejection_reason'] = 'No visible transfer interval beyond optical jitter'
        return record
    lo, hi = max(0, active[0]-1), min(len(present)-1, active[-1]+1)
    while lo > 0 and samples[present[lo]]['kind'] != 'observed':
        lo -= 1
    while hi < len(present)-1 and samples[present[hi]]['kind'] != 'observed':
        hi += 1
    chosen = present[lo:hi+1]
    segment = samples[chosen[0]:chosen[-1]+1]
    missing = sum(s['center_px'] is None for s in segment)
    observed = sum(s['kind'] == 'observed' for s in segment)
    gaps = np.diff([samples[i]['source_frame'] for i in chosen])
    observed_gaps = np.diff([s['source_frame'] for s in segment if s['kind']=='observed'])
    if missing or observed < 6 or max(gaps, default=0) > 10:
        record['rejection_reason'] = 'Insufficient visible transfer coverage or long missing gap'
        return record
    # All transformations are exposed. Endpoints are anchored to reviewed start/end,
    # preserving measured residual shape instead of copying its terminal placement error.
    segment_xy = np.asarray([samples[i]['relative_xy_m'] for i in chosen])
    path = build_path(segment_xy, [samples[i]['source_frame'] for i in chosen])
    by_frame = {s['source_frame']: s for s in segment}
    path['sample_kind'] = ['observed' if by_frame[f]['kind']=='observed' else 'interpolated'
                           for f in path['sample_source_frames']]
    path['sample_weights'] = [max(.3, by_frame[f]['confidence']) if by_frame[f]['kind']=='observed'
                             else .25 for f in path['sample_source_frames']]
    record.update(status='accepted', path=path,
                  start_xy_m=path['relative_xy_m'][0], end_xy_m=path['relative_xy_m'][-1],
                  endpoint_relative_xy_m=[xy[0].tolist(), xy[-1].tolist()],
                  movement_source_frame_start=samples[chosen[0]]['source_frame'],
                  movement_source_frame_terminal=samples[chosen[-1]]['source_frame'],
                  movement_duration_s=(samples[chosen[-1]]['source_pts_s']-samples[chosen[0]]['source_pts_s']),
                  quality={'movement_samples': len(segment), 'observed_movement_samples': observed,
                           'interpolated_movement_samples': len(segment)-observed,
                           'observed_fraction': observed/len(segment),
                           'maximum_sample_gap_frames': int(max(gaps, default=0)),
                           'maximum_observed_bracket_gap_frames': int(max(observed_gaps, default=0)),
                           'confidence_definition': 'heuristic contour evidence; not a probability or metric error bound',
                           'review_status': 'pending visual review of transfer contact sheets'})
    return record


def _write_review(cap, records, output_dir, fps):
    writer = cv2.VideoWriter(str(output_dir/'motion_review.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), 10., (720, 1280))
    for record in records:
        if record['status'] != 'accepted':
            continue
        selected = [s for s in record['samples'] if
                    record['movement_source_frame_start'] <= s['source_frame'] <= record['movement_source_frame_terminal']]
        sheet = Image.new('RGB', (1440, 1080), 'white')
        draw = ImageDraw.Draw(sheet)
        preview_indices = np.linspace(0, len(selected)-1, 12).round().astype(int)
        path = np.asarray([s['center_px'] for s in selected]).round().astype(int)
        for i, item in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, item['source_frame'])
            ok, image = cap.read()
            if not ok:
                raise ValueError('Cannot render motion evidence')
            cv2.polylines(image, [path[:i+1]], False, (255, 120, 0), 2)
            color = (0, 210, 0) if item['kind'] == 'observed' else (0, 180, 255)
            cv2.drawMarker(image, tuple(np.rint(item['center_px']).astype(int)), color,
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(image, f"{record['episode_id']} f{item['source_frame']} {item['kind']}",
                        (8, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 220), 2)
            cv2.putText(image, 'IMAGE XY ONLY; playback resampled; yellow = interpolated',
                        (8, 62), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 220), 1)
            writer.write(image)
            for j in np.flatnonzero(preview_indices == i):
                pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil.thumbnail((240, 500))
                sheet.paste(pil, ((j%6)*240, (j//6)*540+24))
                draw.text(((j%6)*240+3, (j//6)*540+3),
                          f"{record['episode_id']} f{item['source_frame']} {item['kind']}", fill='black')
        sheet.save(output_dir/f"{record['episode_id']}_transfer.jpg")
    writer.release()


def extract_motion(source, episodes_path, output_path, output_dir, stride=2, visual_review_path=None):
    source, output_dir = Path(source), Path(output_dir)
    metadata = json.loads(Path(episodes_path).read_text(encoding='utf-8'))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != metadata['source']['sha256']:
        raise ValueError('Video hash does not match endpoint source lineage')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    visual_review = {}
    if visual_review_path and Path(visual_review_path).exists():
        visual_review = json.loads(Path(visual_review_path).read_text(encoding='utf-8'))
        if visual_review.get('source_sha256') != sha:
            raise ValueError('Visual review belongs to a different source video')
    records = []
    for ep in metadata['episodes']:
        review = visual_review.get('episodes', {}).get(ep['episode_id'], {})
        samples = apply_sample_review(_track_episode(cap, ep, fps, stride), review)
        record = _make_record(ep, samples)
        if review.get('decision') == 'rejected':
            record.update(status='rejected', rejection_reason=review['notes'])
        if record['status']=='accepted' and review.get('decision')=='accepted':
            record['quality'].update(review_status='transfer contact sheet visually reviewed',
                                     review_notes=review['notes'])
        records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_review(cap, records, output_dir, fps)
    cap.release()
    result = {'schema_version': 1, 'source': metadata['source'],
              'endpoint_manifest_sha256': hashlib.sha256(Path(episodes_path).read_bytes()).hexdigest(),
              'visual_review_sha256': hashlib.sha256(Path(visual_review_path).read_bytes()).hexdigest()
                  if visual_review else None,
              'dimensions_m': DIMENSIONS_M,
              'mapping_disclaimer': MAPPING_DISCLAIMER,
              'measurement_limitations': ['Moving center is the visible ring optical footprint or aperture, not a 3D pose.',
                  'Unknown height, tilt, perspective and shadows can bias image-plane metric motion.',
                  'Only short missing gaps are linearly interpolated; confidence zero; no gaps cross edited episode cuts.',
                  'Transfer progress is arc-length normalized; source timing is preserved separately.',
                  'Robot lift height, orientation, grasp contact and actuator commands remain simulation-derived.'],
              'sampling_stride_source_frames': stride, 'episodes': records}
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    load_motion(output_path)
    summary = {'accepted': sum(e['status']=='accepted' for e in records), 'total': len(records),
               'observed_samples': sum(s['kind']=='observed' for e in records for s in e['samples']),
               'interpolated_samples': sum(s['kind']=='short_gap_interpolated' for e in records for s in e['samples']),
               'missing_samples': sum(s['kind']=='missing' for e in records for s in e['samples']),
               'episodes': [{k: e.get(k) for k in ('episode_id','status','rejection_reason','quality')}
                            for e in records]}
    (output_dir/'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/source/359977334.mp4')
    parser.add_argument('--episodes', default='data/real_episodes.json')
    parser.add_argument('--output', default='data/real_motion.json')
    parser.add_argument('--output-dir', default='outputs/motion')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--visual-review', default='outputs/motion/visual_review.json')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    result = extract_motion(args.source, args.episodes, args.output, args.output_dir,
                            args.stride, args.visual_review)
    print(json.dumps({'accepted': sum(e['status']=='accepted' for e in result['episodes']),
                      'total': len(result['episodes'])}))


if __name__ == '__main__':
    main()
