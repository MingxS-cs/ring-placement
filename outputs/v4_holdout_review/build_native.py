import cv2,json,numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
out=Path('outputs/v4_holdout_review'); cap=cv2.VideoCapture('data/source/359977334.mp4')
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',20)
sel={'E20':[2880,2910,2912,2914,2916,2918,2920,2922,2924,2926,2946], 'E21':[2970,3039,3042,3045,3048,3051,3054,3057,3060,3063,3067,3096], 'E22':[3120,3183,3185,3187,3189,3191,3193,3195,3197,3199,3203,3222]}
for eid,frames in sel.items():
 for batch in range(0,len(frames),6):
  chosen=frames[batch:batch+6]
  y0=750 if eid=='E20' else 650 if eid=='E21' else 450
  h=500 if eid!='E22' else 750
  sheet=Image.new('RGB',(1440,(h+36)*3),'white'); d=ImageDraw.Draw(sheet)
  for i,f in enumerate(chosen):
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
   cv2.imwrite(str(out/f'{eid}_f{f}_raw.jpg'),im)
   crop=im[y0:y0+h,:].copy()
   for xx in range(0,720,50):
    cv2.line(crop,(xx,0),(xx,h-1),(100,100,100),1)
    cv2.putText(crop,str(xx),(xx+2,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
   for yy in range(y0,y0+h,50):
    cv2.line(crop,(0,yy-y0),(719,yy-y0),(100,100,100),1)
    cv2.putText(crop,str(yy),(2,yy-y0+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
   tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
   tx,ty=(i%2)*720,(i//2)*(h+36)
   sheet.paste(tile,(tx,ty+36));d.text((tx+5,ty+5),f'{eid} f{f} native crop y={y0}',fill='black',font=font)
  sheet.save(out/f'{eid}_native_{batch//6+1}.png')
cap.release()
