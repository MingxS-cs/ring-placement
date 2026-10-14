"""Review-only artifact builder; coordinates are manually selected visible ring centers."""
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from review_extract import ROOT,OUT,EPISODES,frame,preload_frames,sheet
from ring_learning.motion import build_path,image_relative_xy

POINTS={
 'E03':[(422,115,1004),(424,122,986),(426,136,926),(428,172,854),(430,212,783),(432,244,734),(434,272,701),(436,289,683),(438,294,663),(440,294,650),(474,291,651)],
 'E06':[(876,488,528),(878,485,518),(880,467,510),(882,428,524),(884,383,550),(886,347,575),(888,325,599),(890,317,612),(892,317,632),(894,311,636),(896,307,641),(898,304,646),(900,296,650),(902,293,651),(924,292,652)],
 'E07':[(1010,353,423),(1012,345,423),(1014,335,441),(1016,331,473),(1018,328,516),(1020,327,574),(1022,328,613),(1024,326,635),(1026,316,651),(1028,304,656),(1030,295,661),(1032,293,656),(1056,291,652)],
 'E13':[(1964,668,652),(1966,662,637),(1968,640,594),(1970,606,544),(1972,566,500),(1974,523,459),(1976,478,417),(1978,436,399),(1980,398,390),(1982,376,385),(1984,374,393),(1986,377,397),(1988,380,401),(1990,370,394),(2016,359,399)],
 'E15':[(2252,199,193),(2254,201,183),(2256,210,159),(2258,230,160),(2260,268,194),(2262,317,249),(2264,350,299),(2266,366,336),(2268,363,357),(2270,359,374),(2272,355,381),(2274,359,393),(2276,363,399),(2278,366,399),(2280,365,397),(2298,359,400)],
 'E17':[(2536,132,704),(2538,133,691),(2540,149,645),(2542,196,570),(2544,266,497),(2546,329,429),(2548,368,397),(2550,376,379),(2552,379,368),(2554,367,377),(2556,353,379),(2558,346,389),(2560,342,385),(2580,357,400)],
 'E18':[(2656,298,1060),(2658,302,1020),(2660,317,911),(2662,343,767),(2664,370,620),(2666,386,506),(2668,384,425),(2670,371,384),(2672,363,362),(2674,359,367),(2676,363,380),(2700,360,397)],
 'E19':[(2794,300,91),(2796,299,74),(2798,297,79),(2800,305,92),(2802,322,135),(2804,349,204),(2806,373,273),(2808,379,330),(2810,373,369),(2812,371,397),(2814,366,403),(2816,363,403),(2818,358,400),(2832,358,401)],
 'E11':[(1636,58,841),(1638,70,834),(1640,95,810),(1642,133,767),(1644,186,725),(1646,249,686),(1648,286,662),(1650,307,649),(1652,309,636),(1654,298,634),(1656,292,628),(1658,297,625),(1660,296,626),(1662,301,630),(1698,309,639)],
 'E12':[(1834,274,683),(1836,274,672),(1838,290,626),(1840,310,557),(1842,332,492),(1844,350,445),(1846,358,409),(1848,355,391),(1850,351,388),(1852,354,392),(1854,354,405),(1878,357,399)],
 'E05':[(726,401,818),(728,397,811),(730,386,781),(732,371,736),(734,351,701),(736,332,675),(738,318,655),(740,302,652),(742,290,657),(744,283,666),(746,278,672),(748,285,672),(774,290,667)],
}
BLURRED={'E03':{426,428,430},'E13':{1968,1970,1972,1974,1976},'E15':{2260,2262,2264},'E17':{2540,2542,2544,2546},'E18':{2660,2662,2664,2666},'E19':{2802,2804,2806},'E11':{1640,1642,1644,1646},'E12':{1838,1840,1842},'E05':{730,732,734}}

def write():
    source_sha=hashlib.sha256((ROOT/'data/source/359977334.mp4').read_bytes()).hexdigest()
    lineage=json.loads((OUT/'ordinal_lineage.json').read_text())
    ordinal_map={(r['episode_id'],r['seek_request_frame']):r['actual_decoded_ordinal'] for r in lineage['matches']}
    effective_points={ident:[(ordinal_map[(ident,n)],x,y) for n,x,y in rows] for ident,rows in POINTS.items()}
    effective_blur={ident:{ordinal_map[(ident,n)] for n in rows} for ident,rows in BLURRED.items()}
    wanted={n for rows in effective_points.values() for n,x,y in rows}
    for ident in ('E11','E12'):
        e=EPISODES[ident];wanted.update(range(e['source_frame_start'],e['source_frame_terminal']+1,6))
        for n,x,y in POINTS[ident]:
            old=OUT/'native_overlays'/f'{ident}_{n}.png'
            if old.exists():old.unlink()
    preload_frames(wanted)
    for ident in ('E11','E12'):
        e=EPISODES[ident]
        sheet(ident,list(range(e['source_frame_start'],e['source_frame_terminal']+1,6)),scale=.25,cols=6,suffix='_overview')
        sheet(ident,[n for n,x,y in effective_points[ident]],(0,520,420,920) if ident=='E11' else (190,280,460,740),1,4,'_transfer')
    records=[]
    for ident,points in effective_points.items():
        e=EPISODES[ident]; samples=[]; cells=[]
        for f,x,y in points:
            im,pts=frame(f)
            uncertainty=8 if f in effective_blur.get(ident,set()) else 5
            (OUT/'native_frames').mkdir(exist_ok=True)
            raw_name=f'outputs/v4_development_review/native_frames/{ident}_{f}.png'
            im.save(ROOT/raw_name)
            samples.append(dict(source_frame=f,frame=f,source_pts_s=pts,time=pts,center_px=[x,y],kind='manual_visual',decoded_rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),native_frame=raw_name,visual_uncertainty_px=uncertainty,uncertainty_definition='Qualitative center-selection uncertainty, nominally approximately +/-5 px, increased to +/-8 px for blurred rims; not a calibrated confidence interval or physical measurement error bound.',method='Visible ring rim/aperture center selected manually from sequentially decoded source image; displaced cast shadow excluded.'))
            crop=(max(0,x-100),max(0,y-100),min(720,x+100),min(1280,y+100))
            c=im.crop(crop); d=ImageDraw.Draw(c); cx=x-crop[0];cy=y-crop[1]
            d.ellipse((cx-5,cy-5,cx+5,cy+5),outline='#00ff00',width=2);d.line((cx-13,cy,cx+13,cy),fill='#ff2222',width=2);d.line((cx,cy-13,cx,cy+13),fill='#ff2222',width=2)
            cell=Image.new('RGB',(220,232),'white');cell.paste(c,(10,30));ImageDraw.Draw(cell).text((5,5),f'{ident} f{f} ({x},{y}) PTS={pts:.4f}',fill='black');cells.append(cell)
            (OUT/'native_overlays').mkdir(exist_ok=True)
            d=ImageDraw.Draw(im);d.ellipse((x-5,y-5,x+5,y+5),outline='#00ff00',width=2);d.line((x-13,y,x+13,y),fill='#ff2222',width=2);d.line((x,y-13,x,y+13),fill='#ff2222',width=2)
            im.save(OUT/'native_overlays'/f'{ident}_{f}.png')
        center_sheet=Image.new('RGB',(4*220,((len(cells)+3)//4)*232),'#ddd')
        for i,c in enumerate(cells):center_sheet.paste(c,((i%4)*220,(i//4)*232))
        center_sheet.save(OUT/f'{ident}_center_review.png')
        xy=image_relative_xy([[x,y] for f,x,y in points],e['bearing_center_px'],e['bearing_outer_diameter_px'])
        path=build_path(xy,[f for f,x,y in points]);path['progress_arc']=path['progress']
        records.append(dict(episode_id=ident,scene_group=e['scene_group'],split='val' if ident in ('E05','E11','E12') else 'train',status='accepted',source_frame_start=e['source_frame_start'],source_frame_terminal=e['source_frame_terminal'],bearing_center_px=e['bearing_center_px'],bearing_outer_diameter_px=e['bearing_outer_diameter_px'],movement_source_frame_start=points[0][0],movement_source_frame_terminal=points[-1][0],movement_duration_s=(samples[-1]['source_pts_s']-samples[0]['source_pts_s']),samples=samples,path=path,derived_path=path,start_xy_m=xy[0].tolist(),end_xy_m=xy[-1].tolist(),endpoint_relative_xy_m=xy[-1].tolist(),quality=dict(movement_samples=len(points),observed_movement_samples=len(points),interpolated_movement_samples=0,maximum_sample_gap_frames=max(np.diff([f for f,x,y in points]).tolist()),observed_fraction=1.,review_status='source images and native center overlays visually reviewed',proof_contact_sheet=f'outputs/v4_development_review/{ident}_center_review.png',no_hidden_coordinates=True),notes='Near-pre-lift through visible placement/release inside bearing hole. Visible image-plane center trajectory only; no measured depth, orientation, physical height or calibrated 3D claim.'))
    payload=dict(schema_version=1,source_sha256=source_sha,source=dict(basename='359977334.mp4',sha256=source_sha,frame_definition='zero-based ordinal from sequential decoded source frames, verified by exact decoded pixel hash and actual FFmpeg PTS',pts_provenance='OpenCV FFmpeg CAP_PROP_POS_MSEC after sequential decoded source frame; timestamps are not assumed to equal ordinal/fps',image_dimensions_px=[720,1280]),annotation_policy='Only manually observed visible ring centers. No interpolation or inferred coordinates counted as observations. Original split membership preserved. No fresh holdout or old test episode used.',mapping=dict(method='local bearing diameter normalization',formula='relative_xy_m=[(ru-bu),(bv-rv)]*(0.05/D)',disclaimer='Approximate image-plane XY based on known 50 mm bearing outer diameter; no camera calibration or metric 3D ground truth.'),episodes=records)
    (ROOT/'data/v4_development_annotations.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print([(r['episode_id'],r['split'],len(r['samples'])) for r in records])
if __name__=='__main__':write()
