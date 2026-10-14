import cv2,json,sys,numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
out=Path('outputs/v4_holdout_review')
chosen={'E20':[[2880,236,1070,5],[2910,276,1063,5],[2912,286,1047,8],[2914,327,1012,8],[2916,394,995,8],[2918,451,978,8],[2920,492,974,5],[2922,516,970,5],[2924,527,977,5],[2926,536,979,5],[2946,542,987,5]],'E21':[[2970,49,864,5],[3039,48,863,5],[3042,54,860,5],[3045,106,881,8],[3048,202,914,8],[3051,330,940,8],[3054,444,954,8],[3057,533,959,8],[3060,587,963,5],[3063,600,968,5],[3067,567,979,5],[3096,538,979,5]],'E22':[[3120,614,573,5],[3183,621,572,5],[3185,612,574,5],[3187,593,587,8],[3189,575,643,8],[3191,560,717,8],[3193,550,794,8],[3195,552,861,8],[3197,549,914,8],[3199,548,944,5],[3203,543,982,5],[3222,535,983,5]]}
(out/'manual_visual_centers.json').write_text(json.dumps(chosen,indent=2))
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
for eid,rows in chosen.items():
 sheet=Image.new('RGB',(1020,1440),'white');d=ImageDraw.Draw(sheet)
 for i,(f,x,y,u) in enumerate(rows):
  im=cv2.imread(str(out/f'{eid}_f{f}_raw.jpg'))
  cv2.drawMarker(im,(x,y),(0,255,0),cv2.MARKER_CROSS,24,1)
  cv2.circle(im,(x,y),u,(0,255,255),1)
  cv2.putText(im,f'{eid} f{f}: manual visual center ({x},{y}); approx +/-{u}px',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,220),1)
  cv2.putText(im,'IMAGE PLANE ONLY; no interpolated observations',(8,51),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,220),1)
  cv2.imwrite(str(out/f'{eid}_f{f}_overlay.jpg'),im)
  crop=np.ones((300,300,3),np.uint8)*255
  x0,y0=x-150,y-150;sx0,sy0=max(0,x0),max(0,y0);sx1,sy1=min(720,x+150),min(1280,y+150)
  crop[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=im[sy0:sy1,sx0:sx1]
  tx,ty=(i%3)*340,(i//3)*360;sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(tx,ty+60))
  d.text((tx+3,ty+3),f'{eid} f{f} center {x},{y}',fill='black',font=font)
  d.text((tx+3,ty+27),f'±{u}px; native crop {x0},{y0}',fill='black',font=font)
 sheet.save(out/f'{eid}_manual_visual_review.png')
