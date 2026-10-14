import subprocess,shutil,json,numpy as np,hashlib
from pathlib import Path
x=json.loads(Path('data/v4_holdout_annotations.json').read_text());out=Path('outputs/v4_holdout_review')
cmd=[shutil.which('ffprobe'),'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json','data/source/359977334.mp4']
r=subprocess.run(cmd,capture_output=True,text=True,check=True);p=json.loads(r.stdout);t=np.asarray([float(f['best_effort_timestamp_time']) for f in p['frames']]);assert len(t)==3581
rows=[]
for e in x['episodes']:
 for s in e['samples']:
  f=s['source_frame'];delta=float(t[f]-s['source_pts_s']);assert abs(delta)<5e-7
  rows.append({'source_id':e['source_id'],'source_frame':f,'ffprobe_best_effort_pts_s':float(t[f]),'recorded_pts_s':s['source_pts_s'],'difference_s':delta})
audit={'status':'passed','method':'ffprobe -show_frames from beginning, best_effort_timestamp_time in display-order sequence; checked by zero-based ordinal','source_total_frame_count':len(t),'checked_samples':len(rows),'max_absolute_pts_difference_s':max(abs(r['difference_s']) for r in rows),'samples':rows}
(out/'ffprobe_sequential_pts_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='samples'},indent=2))
