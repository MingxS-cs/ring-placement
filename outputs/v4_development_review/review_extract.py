import json, sys
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/source/359977334.mp4'
EPISODES={x['episode_id']:x for x in json.loads((ROOT/'data/real_episodes.json').read_text())['episodes']}
FRAME_CACHE={}
def preload_frames(frames):
    wanted=set(frames)
    cap=cv2.VideoCapture(str(SOURCE));n=0
    while n<=max(wanted):
        ok,a=cap.read()
        if not ok:raise RuntimeError(n)
        if n in wanted:
            FRAME_CACHE[n]=(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)),cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        n+=1
    cap.release()
def frame(n):
    if n in FRAME_CACHE:
        im,pts=FRAME_CACHE[n]
        return im.copy(),pts
    cap=cv2.VideoCapture(str(SOURCE)); cap.set(cv2.CAP_PROP_POS_FRAMES,n)
    ok, a=cap.read(); pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000; cap.release()
    if not ok: raise RuntimeError(n)
    return Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)),pts
def sheet(identifier, frames, crop=None, scale=.5, cols=4, suffix=''):
    ims=[]
    for n in frames:
        im,pts=frame(n)
        (OUT/'frames').mkdir(exist_ok=True)
        im.save(OUT/'frames'/f'{identifier}_{n}.jpg',quality=95)
        if crop: im=im.crop(crop)
        im=im.resize((round(im.width*scale),round(im.height*scale)))
        cell=Image.new('RGB',(im.width,im.height+28),'white'); cell.paste(im,(0,28))
        ImageDraw.Draw(cell).text((5,5),f'{identifier} f{n} PTS {pts:.4f}s crop={crop}',fill='black')
        ims.append(cell)
    w,h=ims[0].size; result=Image.new('RGB',(cols*w,((len(ims)+cols-1)//cols)*h),'#ddd')
    for i,im in enumerate(ims): result.paste(im,((i%cols)*w,(i//cols)*h))
    result.save(OUT/f'{identifier}{suffix}.jpg',quality=95)
if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    for identifier in sys.argv[1:]:
        e=EPISODES[identifier]
        sheet(identifier,list(range(e['source_frame_start'],e['source_frame_terminal']+1,6)),scale=.25,cols=6,suffix='_overview')
