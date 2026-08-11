# -*- coding: utf-8 -*-
"""GATE 1 (b)+(c): IN-DISTRIBUTION 224 patch-16 CAM (good), refine per-peak CENTER via local CAM centroid (sub-16px,
fixes grid snapping) and SIZE via local CAM extent (vs fixed 42.5px box). Three variants on same peaks:
  fixed          -> reproduce baseline (should be ~0.247/0.098)
  centroid       -> refine center only, keep fixed size
  centroid+extent-> refine center + variable box from CAM blob
Eval vs GT on 60 FASTMAL fields (same protocol as gate1)."""
import os, sys, json, glob
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; TILE=224; P=16; G=14; step=TILE/G  # 16px
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
FIN=json.load(open(OUTPUTS/'experiments'/'s3_filter'/'final.json'))['calib']
gate_thr=FIN['gate_thr']; cam_thr=FIN['cam_thr']; BS=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
SZMIN,SZMAX=24.0,64.0
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
genc=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
genc.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); genc=genc.to(dev).eval()
ghead=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev)
ghead.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); ghead.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
genc.blocks[-4].register_forward_hook(hook)
def gen_logit_cam(crops, bs=16):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        ims=[Image.fromarray(c).resize((TILE,TILE)) for c in crops[i:i+bs]]
        xb=torch.stack([norm(im) for im in ims]).to(dev).requires_grad_(True)
        ff=genc.forward_features(xb); s=ghead(ff['x_norm_clstoken']).squeeze(-1); genc.zero_grad(); ghead.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
def toff(arr):
    H,W=arr.shape[:2]; xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    return [(arr[ty:ty+TILE,tx:tx+TILE],(tx,ty)) for ty in ys for tx in xs]
def refine(sg, gy, gx, R=2, fc=0.3, fe=0.5):
    pv=sg[gy,gx]; i0,i1=max(0,gy-R),min(G,gy+R+1); j0,j1=max(0,gx-R),min(G,gx+R+1)
    win=sg[i0:i1,j0:j1]; ii,jj=np.mgrid[i0:i1,j0:j1]
    wc=np.where(win>=fc*pv, win, 0.0); sw=wc.sum()
    if sw<=0: return (gx+0.5), (gy+0.5), BS, BS
    cr=(wc*ii).sum()/sw; cc=(wc*jj).sum()/sw
    me=win>=fe*pv
    if me.any():
        rr=ii[me]; ccc=jj[me]; hw=(ccc.max()-ccc.min()+1)*step*1.1; hh=(rr.max()-rr.min()+1)*step*1.1
    else: hw=hh=BS
    return (cc+0.5),(cr+0.5),float(np.clip(hw,SZMIN,SZMAX)),float(np.clip(hh,SZMIN,SZMAX))
def pseudo_boxes(arr, mode):
    tiles=toff(arr); crops=[c for c,_ in tiles]; offs=[o for _,o in tiles]
    lo,ca=gen_logit_cam(crops); cand=[]
    for ti,((tx,ty),lg,cm) in enumerate(zip(offs,lo,ca)):
        if lg<gate_thr: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=cam_thr: cand.append((tx+(gx+0.5)*step, ty+(gy+0.5)*step, float(sg[gy,gx]), ti, gy, gx))
    cand.sort(key=lambda z:-z[2]); kept=[]
    for cx,cy,s,ti,gy,gx in cand:
        if all((cx-kx)**2+(cy-ky)**2>=MD*MD for kx,ky,*_ in kept): kept.append((cx,cy,s,ti,gy,gx))
    kept=kept[:300]; boxes=[]; scs=[]
    for cx,cy,s,ti,gy,gx in kept:
        tx,ty=offs[ti]; sg=ca[ti].reshape(G,G)
        if mode=='fixed': boxes.append((cx-BS/2,cy-BS/2,cx+BS/2,cy+BS/2))
        else:
            rc,rr,hw,hh=refine(sg,gy,gx); ncx=tx+rc*step; ncy=ty+rr*step
            if mode=='centroid': boxes.append((ncx-BS/2,ncy-BS/2,ncx+BS/2,ncy+BS/2))
            else: boxes.append((ncx-hw/2,ncy-hh/2,ncx+hw/2,ncy+hh/2))
        scs.append(s)
    return boxes,scs
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,[(x,y,x+w,y+h) for (x,y,w,h) in b]))
    return o
import random; fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields); fields=fields[:60]
print(f'GT fields: {len(fields)}',flush=True)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0
def ap(t,s,n):
    if not t:return 0
    oo=np.argsort(-np.array(s)); t=np.array(t)[oo]; ct=np.cumsum(t);cf=np.cumsum(1-t);rc=ct/max(1,n);pr=ct/np.maximum(1,ct+cf)
    mr=np.r_[0,rc,1];mp=np.r_[0,pr,0]
    for i in range(len(mp)-1,0,-1):mp[i-1]=max(mp[i-1],mp[i])
    k=np.where(mr[1:]!=mr[:-1])[0];return float(np.sum((mr[k+1]-mr[k])*mp[k+1]))
# precompute boxes per field per mode in one CAM pass? modes share peaks but differ only in box shape -> compute all 3 per field
def eval_all():
    res={m:{'at':{0.3:[],0.5:[]},'sc':[],'ng':0} for m in ['fixed','centroid','centroid+extent']}
    for fi,(ip,gt) in enumerate(fields):
        try: arr=np.asarray(Image.open(ip).convert('RGB'))
        except Exception as e: print(f'field {fi} FAIL {e}',flush=True); continue
        for m in res:
            bxs,scs=pseudo_boxes(arr,m); res[m]['ng']+=len(gt); order=np.argsort(-np.array(scs)) if scs else []
            for th in (0.3,0.5):
                mm=set()
                for j in order:
                    bst=-1;bj=-1
                    for jg,g in enumerate(gt):
                        if jg in mm: continue
                        v=iou(bxs[j],g)
                        if v>bst: bst,bj=v,jg
                    tp=1 if bst>=th and bj>=0 else 0
                    if tp:mm.add(bj)
                    res[m]['at'][th].append(tp)
                    if th==0.3: res[m]['sc'].append(scs[j])
        if fi%15==0: print(f'  ...field {fi} done',flush=True)
    for m in res:
        d=res[m]; tot=len(d['at'][0.3]); tp3=sum(d['at'][0.3])
        out=dict(n_pred=tot, precision=round(tp3/max(1,tot),3), recall=round(tp3/max(1,d['ng']),3), **{f'AP@{t}':round(ap(d['at'][t],d['sc'],d['ng']),3) for t in (0.3,0.5)})
        print(f'  {m:16s}: {out}',flush=True)
print('=== GATE 1 (b)+(c): in-distribution centroid/extent refinement ===',flush=True)
eval_all(); print('GATE1C_DONE',flush=True)
