"""Build a sparse, manually reviewed held-out annotation package, not a tracker.

All chosen centers come from the reviewed native image sheets/overlays. The
contour suggestions are review aids only. No missing-frame labels are inferred.
"""
import cv2,hashlib,json,sys,numpy as np
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path.cwd()))
from ring_learning.motion import image_relative_xy,build_path
out=Path('outputs/v4_holdout_review'); source=Path('data/source/359977334.mp4')
manifest=Path('data/real_episodes.json'); metadata=json.loads(manifest.read_text())
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha==metadata['source']['sha256']
labels=json.loads((out/'manual_visual_centers.json').read_text());candidates=json.loads((out/'manual_seed_fit_candidates.json').read_text())
eps={e['episode_id']:e for e in metadata['episodes']}
cap=cv2.VideoCapture(str(source));assert cap.isOpened()
fps=cap.get(cv2.CAP_PROP_FPS)
phase={'E20':(2910,2926),'E21':(3039,3067),'E22':(3183,3203)}
result={'schema_version':1,'purpose':'V4 held-out sparse trajectory evaluation only; do not use for training or model selection',
 'annotation_method':'manual visual center labeling on native source frames, assisted by narrowly gated contour suggestions that were visually reviewed; labels rounded to integer pixels',
 'label_status':'frozen before V4 selected-model freeze; annotation SHA recorded separately in freeze_receipt.json',
 'source':{**metadata['source'],'path':'data/source/359977334.mp4','fps':fps},
 'source_sha256':source_sha,'endpoint_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'existing_motion_sha256_at_annotation':hashlib.sha256(Path('data/real_motion.json').read_bytes()).hexdigest(),
 'annotation_created_utc':datetime.now(timezone.utc).isoformat(),
 'reserved_source_ids':['E20','E21','E22'],
 'holdout_scope':{'freshness':'Fresh trajectory labeling of reserved source windows from an existing phone video, not a new independently collected video.',
  'previous_access':'Initial and terminal endpoint frames/annotations existed before V4. No selected V4 model or predictions were inspected during manual trajectory labeling.',
  'developer_exposure':'Earlier V3 development evaluated E08/E23/E24 repeatedly; those sources are not this fresh trajectory holdout. Original automatic paths of the reserved episodes had been rejected, so their coordinates are not used here.',
  'restriction':'Coordinates, derived path shape, and overlays must remain unread by the training/model-selection agent and root until the selected V4 checkpoint hash is frozen.'},
 'mapping':{'method':'fixed initial local bearing diameter normalization','bearing_outer_diameter_m':0.05,
  'formula':'relative_xy_m = [(u-bu),(bv-v)] * 0.05 / initial_bearing_outer_diameter_px',
  'disclaimer':metadata['mapping']['disclaimer'],
  'pixel_uncertainty_definition':'Qualitative manual localization uncertainty, about +/-5 px clear or +/-8 px blur; not a statistical confidence interval, measured error bound, or total metric-coordinate accuracy.'},
 'measurement_limitations':['Sparse annotations are direct visible optical centers only; unobserved source frames have no measured labels.',
  'A polyline or metric path joining observed anchors is derived for evaluation/visualization and is not an observed continuous trajectory.',
  'Visible annulus/aperture center approximates the ring optical footprint; wall height, blur, tilt and cast shadows can shift it.',
  'Local scale uses the known 50 mm bearing diameter and ignores camera calibration, perspective, distortion, nonuniform scale and height parallax.',
  'Ring height, lift, 3D pose, grasp contact, release time and ground contact/stability remain unvalidated.',
  'Existing initial/terminal labels and the source video were available in earlier development; only the newly reserved transfer labels are blind to V4 model selection.'],
 'boundary_policy':'Every observation stays in the exact endpoint-manifest source-frame window; no label or temporal difference crosses an edited episode cut.',
 'ordinary_time_evaluation_policy':'Evaluate a prediction at the observed source timestamps using source_time_progress or the explicit PTS fields. Arc-length progress is supplied separately and must not replace time for trajectory error. Any interpolation of a prediction or label polyline must be disclosed as derived, not added to the measured sample count.',
 'episodes':[]}
for eid,rows in labels.items():
 e=eps[eid];samples=[];suggestions={r['frame']:r['auto_suggestion'] for r in candidates[eid]}
 for f,x,y,u in rows:
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
  pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  raw=out/f'{eid}_f{f}_raw.jpg';overlay=out/f'{eid}_f{f}_overlay.jpg'
  item={'source_frame':f,'source_pts_s':pts,'episode_time_s':pts-e['source_pts_start_s'],
   'center_px':[x,y],'kind':'manual_visual','confidence':None,
   'pixel_center_uncertainty_px':u,
   'uncertainty_reason':'Motion blur/tilt; center of visible ring annulus/aperture, displaced cast shadow excluded' if u==8 else 'Visible annulus/aperture center; wall projection and minor partial tweezer occlusion',
   'observation_state':'visible','interpolated':False,
   'annotation_provenance':'Native-frame visual review followed by coordinate marker/uncertainty overlay review; not endpoint interpolation or automatic track reuse',
   'endpoint_role':'initial' if f==e['source_frame_start'] else 'terminal' if f==e['source_frame_terminal'] else 'intermediate',
   'evidence':{'raw_image':str(raw).replace('\\','/'),'overlay_image':str(overlay).replace('\\','/'),
    'raw_image_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'overlay_image_sha256':hashlib.sha256(overlay.read_bytes()).hexdigest(),
    'source_image_dimensions_px':[im.shape[1],im.shape[0]]},
   'contour_review_aid':{'used':suggestions[f] is not None,'candidate_center_px':suggestions[f]['center_px'] if suggestions[f] else None,
    'note':'A narrowly gated contour candidate was inspected as a localization aid; final integer label is a manual visual judgment. Missing candidates did not create missing labels when the actual ring remained visible.'}}
  samples.append(item)
 points=image_relative_xy([s['center_px'] for s in samples],e['bearing_center_px'],e['bearing_outer_diameter_px'])
 for s,p in zip(samples,points):s['relative_xy_m']=p.tolist();s['approx_pixel_uncertainty_m']=s['pixel_center_uncertainty_px']*.05/e['bearing_outer_diameter_px']
 path=build_path(points,[s['source_frame'] for s in samples])
 pts=np.asarray([s['source_pts_s'] for s in samples]);pframes=path['sample_source_frames'];byframe={s['source_frame']:s for s in samples}
 path.update(sample_kind=['manual_visual']*len(pframes),sample_weights=[1.]*len(pframes),
  sample_source_pts_s=[byframe[f]['source_pts_s'] for f in pframes],
  source_time_progress=[(byframe[f]['source_pts_s']-pts[0])/(pts[-1]-pts[0]) for f in pframes],
  episode_time_s=[byframe[f]['episode_time_s'] for f in pframes],
  source_time_progress_definition='(decoded source PTS - first observed PTS) / (last observed PTS - first observed PTS); includes visible idle/settling time in the exact episode window',
  measurement_status='Derived planar approximation at observed frames only; no synthesized center samples')
 first_transfer,last_transfer=phase[eid]
 record={'episode_id':eid,'source_id':eid,'scene_group':e['scene_group'],'status':'accepted','rejection_reason':None,
  'source_frame_start':e['source_frame_start'],'source_frame_terminal':e['source_frame_terminal'],
  'candidate_window_end_exclusive_frame':e['candidate_window_end_exclusive_frame'],
  'source_pts_start_s':e['source_pts_start_s'],'source_pts_terminal_s':e['source_pts_terminal_s'],
  'bearing_center_px':e['bearing_center_px'],'bearing_outer_diameter_px':e['bearing_outer_diameter_px'],
  'bearing_reference_provenance':'Previously reviewed initial endpoint bearing ellipse; 50 mm known diameter gives fixed local scale',
  'samples':samples,'path':path,'start_xy_m':points[0].tolist(),'end_xy_m':points[-1].tolist(),
  'endpoint_relative_xy_m':[points[0].tolist(),points[-1].tolist()],
  'evaluation_source_frame_start':e['source_frame_start'],'evaluation_source_frame_terminal':e['source_frame_terminal'],
  'evaluation_duration_s':float(pts[-1]-pts[0]),
  'transfer_review_anchor_interval':{'source_frame_start':first_transfer,'source_frame_terminal':last_transfer,
   'definition':'Visually selected late pre-transfer through late transfer anchors; final endpoint is also retained as a direct observation. These are not exact motion onset, release or ground-contact times.'},
  'release_time_exact_s':None,'ground_contact_verified':False,'continuous_stability_verified':False,
  'quality':{'measured_samples':len(samples),'observed_movement_samples':sum(first_transfer<=s['source_frame']<=last_transfer for s in samples),
   'intermediate_measured_samples':len(samples)-2,'interpolated_samples':0,'missing_labeled_samples':0,
   'minimum_required_intermediate_samples':4,'maximum_source_frame_gap':int(max(np.diff([s['source_frame'] for s in samples]))),
   'review_status':'All chosen native-frame optical centers and uncertainty overlays visually reviewed',
   'coverage':'Sparse visible anchors across transfer; long gaps occur during idle/settling periods and are not filled with measured labels.',
   'review_notes':'Object ring center follows the visible dark annulus/aperture; displaced lighter ring-shaped shadows were excluded. Intermediate coordinates were newly manually labeled, never copied from the rejected automatic track.'},
  'evidence_images':[str(out/f'{eid}_manual_visual_review.png').replace('\\','/')],
  'coordinate_source':'manual visible optical center; approximate fixed initial bearing-local planar normalization'}
 result['episodes'].append(record)
cap.release()
p=Path('data/v4_holdout_annotations.json');p.write_text(json.dumps(result,indent=2),encoding='utf-8')
# This receipt deliberately has counts/IDs/hashes only, so root can audit blindness before model freeze.
receipt={'schema_version':1,'annotation_path':str(p).replace('\\','/'),'annotation_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'source_sha256':source_sha,'endpoint_manifest_sha256':result['endpoint_manifest_sha256'],
 'existing_motion_sha256':result['existing_motion_sha256_at_annotation'],
 'frozen_annotation_created_utc':result['annotation_created_utc'],
 'reserved_source_ids':['E20','E21','E22'],'valid_episodes':3,'measured_samples':35,'interpolated_measured_samples':0,
 'episodes':[{'source_id':r['source_id'],'status':r['status'],'measured_samples':r['quality']['measured_samples'],
  'intermediate_measured_samples':r['quality']['intermediate_measured_samples'],'pixel_uncertainty_px':'approximately 5 clear / 8 blur'} for r in result['episodes']],
 'withheld_until':'Selected V4 model checkpoint hash and selection protocol frozen; root and training agent must not read annotation labels/overlays before then.',
 'scope':'New manual trajectory labels from reserved windows of an existing video; endpoint labels had prior development access; no independent video claim.'}
(out/'freeze_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
