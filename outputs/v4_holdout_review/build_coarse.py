import cv2,json,numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
out=Path('outputs/v4_holdout_review')
x=json.load(open('data/real_episodes.json'))
cap=cv2.VideoCapture('data/source/359977334.mp4')
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
for e in x['episodes']:
 if e['episode_id'] not in ('E20','E21','E22'): continue
 first,last=e['source_frame_start'],e['source_frame_terminal']
 frames=np.linspace(first,last,12).round().astype(int).tolist()
 sheet=Image.new('RGB',(1440,1458),'white'); d=ImageDraw.Draw(sheet)
 for i,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); assert ok
  pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  cv2.imwrite(str(out/f"{e['episode_id']}_f{f}_raw.jpg"),im)
  # Native source coordinates; crop covers transfer, retaining no tracking overlays.
  crop=im[300:1200,:].copy()
  for xx in range(0,720,100):
   cv2.line(crop,(xx,0),(xx,899),(100,100,100),1)
   cv2.putText(crop,str(xx),(xx+3,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
  for yy in range(400,1200,100):
   cv2.line(crop,(0,yy-300),(719,yy-300),(100,100,100),1)
   cv2.putText(crop,str(yy),(3,yy-300-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
  tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((360,450))
  tx,ty=(i%4)*360,(i//4)*486
  sheet.paste(tile,(tx,ty+36));d.text((tx+5,ty+5),f"{e['episode_id']} f{f} PTS {pts:.3f}s",fill='black',font=font)
 sheet.save(out/f"{e['episode_id']}_coarse_contact.png")
cap.release()
