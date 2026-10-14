import cv2
p='data/source/359977334.mp4';targets={422,876,1010,1636,1834,1964,2252,2536,2656,2794}
cap=cv2.VideoCapture(p);out=[];i=0
while i<=max(targets):
    ok,a=cap.read()
    if i in targets:
        pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        b=cv2.VideoCapture(p);b.set(cv2.CAP_PROP_POS_FRAMES,i);ok,c=b.read()
        out.append((i,pts,b.get(cv2.CAP_PROP_POS_MSEC)/1000,bool((a==c).all())))
        b.release()
    i+=1
print(out)
