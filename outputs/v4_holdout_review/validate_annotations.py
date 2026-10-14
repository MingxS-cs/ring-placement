import cv2,hashlib,json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from ring_learning.motion import load_motion,image_relative_xy,build_path
out=Path('outputs/v4_holdout_review');p=Path('data/v4_holdout_annotations.json')
x=json.loads(p.read_text());r=json.loads((out/'freeze_receipt.json').read_text());m=json.loads(Path('data/real_episodes.json').read_text());eps={e['episode_id']:e for e in m['episodes']}
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['annotation_sha256']
assert hashlib.sha256(Path('data/source/359977334.mp4').read_bytes()).hexdigest()==x['source_sha256']==m['source']['sha256']
assert hashlib.sha256(Path('data/real_episodes.json').read_bytes()).hexdigest()==x['endpoint_manifest_sha256']
assert hashlib.sha256(Path('data/real_motion.json').read_bytes()).hexdigest()==x['existing_motion_sha256_at_annotation']
assert set(load_motion(p))=={'E20','E21','E22'}
cap=cv2.VideoCapture('data/source/359977334.mp4');assert cap.isOpened()
counts=[]
for e in x['episodes']:
 assert e['scene_group']==3 and e['status']=='accepted'
 old=eps[e['episode_id']]
 assert e['source_frame_start']==old['source_frame_start'] and e['source_frame_terminal']==old['source_frame_terminal']
 s=e['samples'];frames=np.asarray([z['source_frame'] for z in s]);pts=np.asarray([z['source_pts_s'] for z in s]);xy=np.asarray([z['center_px'] for z in s]);unc=np.asarray([z['pixel_center_uncertainty_px'] for z in s])
 assert 6<=len(s)<=12 and len(s)-2>=4
 assert (frames>=old['source_frame_start']).all() and (frames<=old['source_frame_terminal']).all() and (np.diff(frames)>0).all()
 assert frames[0]==old['source_frame_start'] and frames[-1]==old['source_frame_terminal']
 assert (np.diff(pts)>0).all() and np.isfinite(xy).all() and ((xy[:,0]>=0)&(xy[:,0]<720)&(xy[:,1]>=0)&(xy[:,1]<1280)).all()
 assert set(unc.tolist()).issubset({5,8})
 assert all(z['kind']=='manual_visual' and z['observation_state']=='visible' and z['interpolated'] is False for z in s)
 assert e['quality']['interpolated_samples']==0
 expected=image_relative_xy(xy,e['bearing_center_px'],e['bearing_outer_diameter_px'])
 assert np.allclose(expected,np.asarray([z['relative_xy_m'] for z in s]),atol=1e-12)
 assert np.allclose(e['path']['relative_xy_m'],expected,atol=1e-12)
 assert np.allclose(e['path']['source_time_progress'],(pts-pts[0])/(pts[-1]-pts[0]),atol=1e-12)
 for z in s:
  cap.set(cv2.CAP_PROP_POS_FRAMES,z['source_frame']);ok,im=cap.read();assert ok
  actual_pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  assert abs(actual_pts-z['source_pts_s'])<1e-8
  for typ in ('raw','overlay'):
   img=Path(z['evidence'][typ+'_image']);assert img.exists() and cv2.imread(str(img)).shape[:2]==(1280,720)
   assert hashlib.sha256(img.read_bytes()).hexdigest()==z['evidence'][typ+'_image_sha256']
 counts.append({'source_id':e['source_id'],'measured_samples':len(s),'intermediate_measured_samples':len(s)-2})
cap.release()
report={'status':'passed','checks':['source SHA match','endpoint metadata and existing motion unchanged','exact episode boundaries and monotonic PTS','decoded PTS independently rechecked','6-12 direct observations per episode and >=4 intermediate observations','no interpolated measured labels','finite native-resolution image centers','all evidence image hashes/dimensions','50 mm bearing-relative normalization','arc/path consistency via existing load_motion','source-time progress consistency','annotation freeze SHA match'],'episodes':counts,'annotation_sha256':r['annotation_sha256']}
(out/'validation_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
