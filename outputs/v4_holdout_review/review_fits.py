import cv2,json,sys,numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from ring_learning.motion import detect_moving_ring
out=Path('outputs/v4_holdout_review')
seeds={'E20':[[2880,235,1070],[2910,274,1067],[2912,286,1044],[2914,333,1010],[2916,393,977],[2918,449,960],[2920,494,970],[2922,515,967],[2924,527,977],[2926,536,979],[2946,542,987]],'E21':[[2970,49,864],[3039,49,866],[3042,55,858],[3045,106,879],[3048,202,904],[3051,330,926],[3054,447,944],[3057,540,946],[3060,590,955],[3063,603,963],[3067,572,979],[3096,538,979]],'E22':[[3120,614,573],[3183,621,565],[3185,612,568],[3187,595,585],[3189,575,635],[3191,561,700],[3193,550,795],[3195,550,866],[3197,549,910],[3199,548,944],[3203,544,986],[3222,535,983]]}
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
eps={e['episode_id']:e for e in json.load(open('data/real_episodes.json'))['episodes']}
result={}
for eid,rows in seeds.items():
 sheet=Image.new('RGB',(1040,870),'white');d=ImageDraw.Draw(sheet);result[eid]=[]
 for i,(f,x,y) in enumerate(rows):
  im=cv2.imread(str(out/f'{eid}_f{f}_raw.jpg'))
  det=detect_moving_ring(im,[x,y],eps[eid]['bearing_outer_diameter_px'],22)
  result[eid].append({'frame':f,'manual_seed':[x,y],'auto_suggestion':det})
  crop=np.ones((240,240,3),np.uint8)*255
  x0,y0=x-120,y-120
  sx0,sy0=max(0,x0),max(0,y0);sx1,sy1=min(720,x+120),min(1280,y+120)
  crop[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=im[sy0:sy1,sx0:sx1]
  cv2.drawMarker(crop,(120,120),(0,0,255),cv2.MARKER_CROSS,16,1)
  if det:
   xy=(np.asarray(det['center_px'])-[x0,y0]).round().astype(int)
   cv2.drawMarker(crop,tuple(xy),(0,255,0),cv2.MARKER_CROSS,24,1)
   center=det['center_px'];axes=det['ellipse_axes_px'];cv2.ellipse(crop,(tuple(xy.astype(float)),tuple(axes),det['ellipse_angle_deg']),(0,255,0),1)
  tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
  tx,ty=(i%4)*260,(i//4)*290;sheet.paste(tile,(tx,ty+50));
  d.text((tx+3,ty+2),f'{eid} f{f} seed {x},{y}',fill='black',font=font)
  d.text((tx+3,ty+24), 'fit '+(','.join(f'{v:.1f}' for v in det['center_px']) if det else 'none'),fill='black',font=font)
 sheet.save(out/f'{eid}_fit_roi_review.png')
(out/'manual_seed_fit_candidates.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:[(r['frame'],r['auto_suggestion']['center_px'] if r['auto_suggestion'] else None) for r in v] for k,v in result.items()},indent=2))
