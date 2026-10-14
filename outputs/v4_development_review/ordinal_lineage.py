import cv2,json,hashlib
from pathlib import Path
from write_annotations import POINTS

src='data/source/359977334.mp4'
wanted={n+d for rows in POINTS.values() for n,x,y in rows for d in (-2,-1,0,1,2)}
frames={};cap=cv2.VideoCapture(src);n=0
while n<=max(wanted):
    ok,a=cap.read()
    if not ok:raise RuntimeError(n)
    if n in wanted:frames[n]=(hashlib.sha256(a.tobytes()).hexdigest(),cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    n+=1
cap.release();matches=[]
for ident,rows in POINTS.items():
    for n,x,y in rows:
        cap=cv2.VideoCapture(src);cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,a=cap.read()
        digest=hashlib.sha256(a.tobytes()).hexdigest();pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;cap.release()
        candidates=[i for i in range(n-2,n+3) if frames[i][0]==digest and abs(frames[i][1]-pts)<1e-8]
        assert len(candidates)==1,(ident,n,candidates)
        matches.append(dict(episode_id=ident,seek_request_frame=n,actual_decoded_ordinal=candidates[0],source_pts_s=pts))
report={'method':'Source sequential decode frame ordinal; exact decoded pixel hash AND actual FFmpeg PTS match to seeked review image.','matches':matches}
Path('outputs/v4_development_review/ordinal_lineage.json').write_text(json.dumps(report,indent=2)+'\n')
print([(i['episode_id'],i['seek_request_frame'],i['actual_decoded_ordinal']) for i in matches if i['seek_request_frame']!=i['actual_decoded_ordinal']])
