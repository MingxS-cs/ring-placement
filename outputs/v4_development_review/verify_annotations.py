import json,hashlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ring_learning.motion import load_motion,image_relative_xy

p=Path('data/v4_development_annotations.json')
d=json.loads(p.read_text()); m=load_motion(p)
allowed={'E03','E06','E07','E13','E15','E17','E18','E19','E05','E11','E12'}
assert set(m)==allowed
assert d['source_sha256']==hashlib.sha256(Path('data/source/359977334.mp4').read_bytes()).hexdigest()
count=0;rows=[]
episodes={x['episode_id']:x for x in json.loads(Path('data/real_episodes.json').read_text())['episodes']}
expected_val={'E05','E11','E12'}
lookup={s['source_frame']:s for r in m.values() for s in r['samples']}
cap=cv2.VideoCapture('data/source/359977334.mp4');n=0
while n<=max(lookup):
    ok,a=cap.read();assert ok
    if n in lookup:
        s=lookup[n]
        assert abs(s['source_pts_s']-cap.get(cv2.CAP_PROP_POS_MSEC)/1000)<1e-8
        assert hashlib.sha256(cv2.cvtColor(a,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==s['decoded_rgb_sha256']
        assert hashlib.sha256(Image.open(s['native_frame']).tobytes()).hexdigest()==s['decoded_rgb_sha256']
    n+=1
cap.release()
for ident,r in m.items():
    e=episodes[ident];ss=r['samples']
    assert r['split']==('val' if ident in expected_val else 'train')
    assert all(s['kind']=='manual_visual' for s in ss)
    assert all(e['source_frame_start']<=s['source_frame']<=e['source_frame_terminal'] for s in ss)
    assert all(a['source_pts_s']<b['source_pts_s'] for a,b in zip(ss,ss[1:]))
    assert all(0<=s['center_px'][0]<720 and 0<=s['center_px'][1]<1280 for s in ss)
    assert len(ss)>=10
    xy=image_relative_xy([s['center_px'] for s in ss],r['bearing_center_px'],r['bearing_outer_diameter_px'])
    assert np.allclose(xy,r['path']['relative_xy_m'])
    assert np.linalg.norm(xy[-1])<0.0075
    assert Path(r['quality']['proof_contact_sheet']).is_file()
    count+=len(ss)
    rows.append({'episode_id':ident,'split':r['split'],'genuine_samples':len(ss),'final_offset_mm':round(float(np.linalg.norm(xy[-1])*1000),2),'arc_length_mm':round(r['path']['arc_length_m']*1000,2)})
report={'status':'passed','source_sha256':d['source_sha256'],'annotation_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'genuine_samples':count,'train_episodes':sum(r['split']=='train' for r in m.values()),'val_episodes':sum(r['split']=='val' for r in m.values()),'checks':['source hash','strict in-episode frames','sequential decoded source frame PTS and exact RGB pixel hash','native source PNG exact decoded pixels','manual visible sample kind','original split membership','image bounds','path reconstruction','clear inside-hole terminal offset','contact proof exists'],'episodes':rows}
Path('outputs/v4_development_review/verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
