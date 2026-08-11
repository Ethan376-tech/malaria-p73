# -*- coding: utf-8 -*-
"""Faintness multi-signal PROTOTYPE v2: per-box extraction with box coords, field_id, and BETTER physical faintness.
Adds vs v1: cx,cy,field_id  +  optical-density (Beer-Lambert) physical measures on the box CENTRE vs local background:
  od_peak (peak chromatin OD), od_integ (integrated OD), blob_area (fraction clearly stained), purple (Giemsa chroma).
Keeps: p_mean,p_var (MC T=30), tta_var (6 views), emb[384], weber (v1 metric, for comparison), score, is_tp, split.
Saves outputs/experiments/s3_filter/perbox_multi2.npz.  Run: /root/miniconda3/bin/python dev_faint_multi_extract2.py"""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
from sklearn.metrics import roc_curve
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=30; BS=24
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
RES=OUTPUTS/'experiments'/'s3_filter'
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
from dinov2.models.vision_transformer import vit_small
def td():
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'encoder.pth',map_location='cpu'),strict=False); return m
enc=td().to(dev); head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1)).to(dev)
head.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_thickdino_ft_ibadan'/'head.pth',map_location='cpu')); enc.eval(); head.eval()
acts={}
def hook(mod,i,o):
    if o.requires_grad: o.retain_grad()
    acts['o']=o
enc.blocks[-4].register_forward_hook(hook)
def logit_cam(crops,bs=BS):
    lo=[];ca=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev).requires_grad_(True)
        ff=enc.forward_features(xb); s=head(ff['x_norm_clstoken']).squeeze(-1); enc.zero_grad(); head.zero_grad(); s.sum().backward()
        o=acts['o']; al=o.grad[:,1:,:].mean(1,keepdim=True); cam=F.relu((o[:,1:,:].detach()*al).sum(-1))
        lo.append(s.detach().cpu().numpy()); ca.append(cam.detach().cpu().numpy())
    return np.concatenate(lo),np.concatenate(ca)
@torch.no_grad()
def mc(crops):
    for mod in list(enc.modules())+list(head.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in enc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    xb=torch.stack([norm(Image.fromarray(c)) for c in crops]).to(dev); pr=[]
    for _ in range(T_MC): ff=enc.forward_features(xb); pr.append(torch.sigmoid(head(ff['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
    enc.eval(); head.eval(); A=np.stack(pr); return A.mean(0),A.var(0)
@torch.no_grad()
def embed_and_tta(crops,bs=BS):
    enc.eval(); head.eval(); EMB=[]; TV=[]
    for i in range(0,len(crops),bs):
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev)
        ff=enc.forward_features(xb); EMB.append(ff['x_norm_clstoken'].cpu().numpy())
        views=[xb, torch.flip(xb,[3]), torch.flip(xb,[2]), torch.rot90(xb,1,[2,3]), torch.rot90(xb,2,[2,3]), torch.rot90(xb,3,[2,3])]
        pv=[torch.sigmoid(head(enc.forward_features(v)['x_norm_clstoken']).squeeze(-1)).cpu().numpy() for v in views]
        TV.append(np.stack(pv).var(0))
    return np.concatenate(EMB), np.concatenate(TV)
def phys(arr,cx,cy,bs):
    """physical faintness via Beer-Lambert optical density at the box centre vs local (bright) background."""
    H,W=arr.shape[:2]; h=bs/2
    x0=int(max(cx-h,0));x1=int(min(cx+h,W));y0=int(max(cy-h,0));y1=int(min(cy+h,H))
    box=arr[y0:y1,x0:x1].astype(np.float32)
    if box.size==0 or box.shape[0]<4 or box.shape[1]<4: return 0.,0.,0.,0.,0.
    lum=box.mean(2); hh=box.shape[0]//4; ww=box.shape[1]//4
    inner=box[hh:box.shape[0]-hh, ww:box.shape[1]-ww]; inlum=inner.mean(2)
    bg=float(np.quantile(lum,0.8))                              # film background (light)
    Imin=float(inlum.min())
    weber=(bg-Imin)/(bg+1.0)                                    # v1-style
    od_peak=float(np.log((bg+1)/(Imin+1)))                      # peak chromatin OD (high=dark parasite)
    od=np.clip(np.log((bg+1)/(inlum+1)),0,None); od_integ=float(od.mean())
    blob_area=float((od>0.15).mean())                          # fraction of centre clearly stained
    odc=np.clip(np.log((bg+1)/(inner+1)),0,None)
    purple=float(((odc[...,0]+odc[...,2])/2 - odc[...,1]).max())# Giemsa purple chroma (R,B high, G low)
    return weber,od_peak,od_integ,blob_area,purple
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['width']),float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,b))
    return o
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    offs=[(tx,ty) for ty in ys for tx in xs]; return [arr[ty:ty+TILE,tx:tx+TILE] for (tx,ty) in offs],offs,arr
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def nms(c,md):
    c=sorted(c,key=lambda z:-z[2]); k=[]
    for cx,cy,s in c:
        if all((cx-kx)**2+(cy-ky)**2>=md*md for kx,ky,_ in k): k.append((cx,cy,s))
    return k
fields=sorted(fastmal_fields(),key=lambda z:str(z[0])); random.Random(0).shuffle(fields)
devF=fields[0::2]; testF=fields[1::2]; bs=CAL['box_size_px']; md=CAL['peak_min_dist_px']
print(f'fields total {len(fields)} dev {len(devF)} test {len(testF)} box {bs:.1f} md {md:.1f}',flush=True)
gl=[];gh=[];cv=[];cl=[]
for ip,gtb in devF[:20]:
    try: im=Image.open(ip).convert('RGB')
    except Exception: continue
    cents=[(x+w/2,y+h/2) for (x,y,w,h) in gtb]; crops,offs,_=toff(im); lo,ca=logit_cam(crops)
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        has=any(tx<=cx<tx+TILE and ty<=cy<ty+TILE for (cx,cy) in cents); gl.append(lg); gh.append(1 if has else 0)
        if has:
            for gy in range(G):
                for gx in range(G):
                    px,py=tx+gx*P+P/2,ty+gy*P+P/2; cv.append(cm[gy*G+gx]); cl.append(1 if any((px-cx)**2+(py-cy)**2<=(P*1.5)**2 for (cx,cy) in cents) else 0)
f,t,th=roc_curve(gh,gl); gate_thr=float(th[np.argmax(t-f)]); f2,t2,th2=roc_curve(cl,cv); cam_thr=float(th2[np.argmax(t2-f2)])
print(f'gate_thr {gate_thr:.3f} cam_thr {cam_thr:.3f}',flush=True)
KEYS=['p_mean','p_var','tta_var','score','is_tp','weber','od_peak','od_integ','blob_area','purple','cx','cy','field_id','split']
def gen(fset,split,fid0):
    R={k:[] for k in KEYS}; EMB=[]; n_gt=0; fid=fid0
    for fi,(ip,gtb) in enumerate(fset):
        try: im=Image.open(ip).convert('RGB')
        except Exception: fid+=1; continue
        gt=[(x,y,x+w,y+h) for (x,y,w,h) in gtb]; n_gt+=len(gt); crops,offs,arr=toff(im); lo,ca=logit_cam(crops); cand=[]
        for (tx,ty),lg,cm in zip(offs,lo,ca):
            if lg<gate_thr: continue
            sg=cm.reshape(G,G)
            for gy in range(G):
                for gx in range(G):
                    if sg[gy,gx]>=cam_thr: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
        peaks=nms(cand,md)
        if not peaks: fid+=1; continue
        H,W=arr.shape[:2]; bc=[]
        for (cx,cy,s) in peaks:
            x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE)); bc.append(arr[y0:y0+TILE,x0:x0+TILE])
        pm,pv=mc(bc); emb,tv=embed_and_tta(bc)
        boxes=[(cx-bs/2,cy-bs/2,cx+bs/2,cy+bs/2) for (cx,cy,s) in peaks]
        order=sorted(range(len(peaks)),key=lambda j:-peaks[j][2]); matched=set(); tpflag=[0]*len(peaks)
        for j in order:
            b=boxes[j]; best=-1;bj=-1
            for jg,g in enumerate(gt):
                if jg in matched: continue
                v=iou(b,g)
                if v>best: best,bj=v,jg
            if best>=0.3 and bj>=0: tpflag[j]=1; matched.add(bj)
        for j,(cx,cy,s) in enumerate(peaks):
            wb,odp,odi,ba,pu=phys(arr,cx,cy,bs)
            vals=[float(pm[j]),float(pv[j]),float(tv[j]),float(s),int(tpflag[j]),wb,odp,odi,ba,pu,float(cx),float(cy),fid,split]
            for k,v in zip(KEYS,vals): R[k].append(v)
            EMB.append(emb[j])
        fid+=1
        if (fi+1)%25==0: print(f'  {"dev" if split==0 else "test"} {fi+1}/{len(fset)} boxes {len(R["is_tp"])}',flush=True)
    return R,np.array(EMB,dtype=np.float32),n_gt,fid
Rd,Ed,ngt_d,fid1=gen(devF,0,0); Rt,Et,ngt_t,_=gen(testF,1,fid1)
out={k:np.array(Rd[k]+Rt[k]) for k in KEYS}; EMB=np.concatenate([Ed,Et],0)
np.savez(RES/'perbox_multi2.npz', emb=EMB, n_gt_dev=ngt_d, n_gt_test=ngt_t, gate_thr=gate_thr, cam_thr=cam_thr, box_size=bs, **out)
dd=out['split']==0; tt=out['split']==1
print(f'\nDEV {int(dd.sum())} (TP {int(out["is_tp"][dd].sum())}) TEST {int(tt.sum())} (TP {int(out["is_tp"][tt].sum())}); fields dev {fid1} EMB {EMB.shape}')
print('MULTI_EXTRACT2_DONE',flush=True)
