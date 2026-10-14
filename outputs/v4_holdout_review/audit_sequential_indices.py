import cv2,json,hashlib
from pathlib import Path
p=Path('data/v4_holdout_annotations.json');x=json.loads(p.read_text());out=Path('outputs/v4_holdout_review')
rows=[(e['episode_id'],s) for e in x['episodes'] for s in e['samples']]
wanted=set()
for eid,s in rows:wanted.update(range(s['source_frame']-2,s['source_frame']+3))
cap=cv2.VideoCapture('data/source/359977334.mp4');assert cap.isOpened()
seq={};n=0
while True:
 ok,im=cap.read()
 if not ok:break
 if n in wanted:
  ok,buf=cv2.imencode('.jpg',im);assert ok
  seq[n]={'jpeg_sha256':hashlib.sha256(buf.tobytes()).hexdigest(),'source_pts_s':cap.get(cv2.CAP_PROP_POS_MSEC)/1000}
 n+=1
cap.release()
report=[]
for eid,s in rows:
 f=s['source_frame'];h=s['evidence']['raw_image_sha256']
 matched=[k for k in range(f-2,f+3) if seq[k]['jpeg_sha256']==h]
 report.append({'source_id':eid,'requested_frame':f,'requested_pts_s':s['source_pts_s'],'matching_sequential_ordinals':matched,'sequential_pts_s_at_requested_ordinal':seq[f]['source_pts_s'],'matching_pts_s':[seq[k]['source_pts_s'] for k in matched]})
audit={'source_total_sequential_frames':n,'definition':'Sequential zero-based decoded frame ordinal counted from first read, compared by re-encoding each decoded frame with default cv2 JPEG settings identical to raw evidence files. No coordinates or images in this audit.','samples':report}
(out/'sequential_index_audit_before_correction.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
