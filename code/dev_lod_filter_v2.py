# -*- coding: utf-8 -*-
"""filter-LoD, EFFICIENT rewrite (v2). Same science as dev_lod_filter.py — does the faintness-protected filter
lower the Mehanian per-µL LoD, on the CAM pseudo-box detector (UNFILTERED all peaks vs FILTERED single-signal
keep: sigma^2_MC>=b* OR p_mean>=a*)? WBC denominator reused from RetinaNet (lod_meh_perwb.npy).
v1 STALLED on high-parasitaemia samples; v2 fixes the three causes: (1) NMS was O(n^2) -> grid-based O(n);
(2) mc() stacked ALL peaks of a field into one batch and ran 30 forwards on it (thousands of images -> GPU
memory thrash) -> mc() now BATCHES (BS=48); (3) MC ran on every peak -> now CAPPED to the top-CAP peaks by CAM
score per field (negatives, which set the LoD, are sparse so never capped; only dense POSITIVE fields, used for
band-sensitivity only, are approximated: beyond-cap peaks are counted as kept).
Output -> outputs/experiments/lod_mehanian/results_filter.json"""
import os, sys, json, glob, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
import pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHECKPOINTS, OUTPUTS
dev='cuda'; TILE=224; P=16; G=TILE//P; T_MC=30; BS=48; CAP=800   # BS: MC batch size; CAP: max peaks/field for MC
random.seed(0); np.random.seed(0); torch.manual_seed(0)
ASTAR=0.1082; BSTAR=0.000224                       # single-signal dev operating point (keep if var>=b* OR mean>=a*)
KMAX=15; KS=[6,15]; Z=1.645
CAL=json.load(open(OUTPUTS/'experiments'/'s2_calib'/'calib_constants.json'))['A']
BOX=CAL['box_size_px']; MD=CAL['peak_min_dist_px']
z=np.load(OUTPUTS/'experiments'/'s3_filter'/'perbox_multi2.npz'); GATE=float(z['gate_thr']); CAMT=float(z['cam_thr'])
RES=OUTPUTS/'experiments'/'lod_mehanian'; RES.mkdir(parents=True,exist_ok=True)
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
def mc(crops,bs=BS):
    for mod in list(enc.modules())+list(head.modules()):
        if isinstance(mod,nn.Dropout): mod.train()
    for mod in enc.modules():
        if mod.__class__.__name__=='DropPath': mod.train()
    means=[]; vars=[]
    for i in range(0,len(crops),bs):                                    # BATCHED (v1 stacked all -> thrash)
        xb=torch.stack([norm(Image.fromarray(c)) for c in crops[i:i+bs]]).to(dev); pr=[]
        for _ in range(T_MC): pr.append(torch.sigmoid(head(enc.forward_features(xb)['x_norm_clstoken']).squeeze(-1)).cpu().numpy())
        A=np.stack(pr); means.append(A.mean(0)); vars.append(A.var(0))
    enc.eval(); head.eval(); return np.concatenate(means), np.concatenate(vars)
def toff(im):
    W,H=im.size; arr=np.asarray(im); xs=sorted(set(list(range(0,max(1,W-TILE+1),TILE))+[max(0,W-TILE)])); ys=sorted(set(list(range(0,max(1,H-TILE+1),TILE))+[max(0,H-TILE)]))
    offs=[(tx,ty) for ty in ys for tx in xs]; return [arr[ty:ty+TILE,tx:tx+TILE] for (tx,ty) in offs],offs,arr
def nms_grid(cand,md):                                                  # O(n): bin to grid, keep highest per cell, check 3x3
    cand=sorted(cand,key=lambda z:-z[2]); cell={}; kept=[]
    for cx,cy,s in cand:
        gx,gy=int(cx//md),int(cy//md); ok=True
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                p=cell.get((gx+dx,gy+dy))
                if p is not None and (cx-p[0])**2+(cy-p[1])**2<md*md: ok=False; break
            if not ok: break
        if ok: cell[(gx,gy)]=(cx,cy); kept.append((cx,cy,s))
    return kept
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3]); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); it=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it; return it/ua if ua>0 else 0.0
def detect(im):
    """-> (peaks[top-CAP], keep-mask over those, n_all_total, arr)."""
    crops,offs,arr=toff(im); lo,ca=logit_cam(crops); cand=[]
    for (tx,ty),lg,cm in zip(offs,lo,ca):
        if lg<GATE: continue
        sg=cm.reshape(G,G)
        for gy in range(G):
            for gx in range(G):
                if sg[gy,gx]>=CAMT: cand.append((tx+gx*P+P/2,ty+gy*P+P/2,float(sg[gy,gx])))
    peaks=nms_grid(cand,MD); n_all=len(peaks)
    if n_all==0: return [],np.array([],bool),0,arr
    peaks=sorted(peaks,key=lambda z:-z[2])[:CAP]                        # cap for MC (top CAM score)
    H,W=arr.shape[:2]; bc=[]
    for (cx,cy,s) in peaks:
        x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE)); bc.append(arr[y0:y0+TILE,x0:x0+TILE])
    pm,pv=mc(bc); keep=(pv>=BSTAR)|(pm>=ASTAR)
    return peaks,keep,n_all,arr
def fastmal_par():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; o=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            b=[(float(r['x']),float(r['y']),float(r['x'])+float(r['width']),float(r['y'])+float(r['height'])) for r in fld['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r.get('type','')]
            if b: o.append((ip,b))
    return o
def re_pr():
    fields=fastmal_par(); tpA=fpA=tpF=fpF=NGT=0
    for n,(ip,gt) in enumerate(fields):
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        peaks,keep,n_all,_=detect(im); NGT+=len(gt)
        boxes=[(cx-BOX/2,cy-BOX/2,cx+BOX/2,cy+BOX/2) for (cx,cy,s) in peaks]
        order=sorted(range(len(peaks)),key=lambda j:-peaks[j][2])
        for tag,sel in (('A',np.ones(len(peaks),bool)),('F',keep if len(peaks) else np.array([],bool))):
            matched=set()
            for j in order:
                if not sel[j]: continue
                b=boxes[j]; best=-1;bj=-1
                for jg,g in enumerate(gt):
                    if jg in matched: continue
                    v=iou(b,g)
                    if v>best: best,bj=v,jg
                if best>=0.3 and bj>=0:
                    matched.add(bj)
                    if tag=='A': tpA+=1
                    else: tpF+=1
                else:
                    if tag=='A': fpA+=1
                    else: fpF+=1
        if (n+1)%40==0: print(f'  re_pr {n+1}/{len(fields)}',flush=True)
    reA=tpA/max(1,NGT); prA=tpA/max(1,tpA+fpA); reF=tpF/max(1,NGT); prF=tpF/max(1,tpF+fpF)
    return dict(re_all=reA,pr_all=prA,re_filt=reF,pr_filt=prF,ngt=NGT,tpA=tpA,fpA=fpA,tpF=tpF,fpF=fpF)
print('=== CAM detector recall/precision on FASTMAL PARASITE fields (unfiltered vs filtered) ===',flush=True)
RP=re_pr()
print(f'UNFILT recall {RP["re_all"]:.3f} precision {RP["pr_all"]:.3f} (TP {RP["tpA"]} FP {RP["fpA"]} GT {RP["ngt"]})',flush=True)
print(f'FILT   recall {RP["re_filt"]:.3f} precision {RP["pr_filt"]:.3f} (TP {RP["tpF"]} FP {RP["fpF"]} GT {RP["ngt"]})',flush=True)
sp=pd.read_csv(OUTPUTS.parent/'data'/'splits'/'splits.csv'); ib=sp[sp.dataset=='ibadan'].reset_index(drop=True)
per_wb=np.load('/root/lod_meh_perwb.npy',allow_pickle=True)
ppul=ib.parasites_per_ul.values.astype(float); pcnt=ib.parasite_count.values.astype(float)
mp_all=[]; mp_filt=[]
print(f'\n=== CAM MP counting (all & filtered) over {KMAX} fields for {len(ib)} samples ===',flush=True)
for k,r in ib.iterrows():
    fields=sorted(glob.glob(str(RAW/r.rel_path)+'/*.tiff')); random.Random(k).shuffle(fields); fields=fields[:KMAX]
    ca=[]; cf=[]
    for f in fields:
        try: im=Image.open(f).convert('RGB')
        except Exception: ca.append(0); cf.append(0); continue
        peaks,keep,n_all,_=detect(im)
        ca.append(n_all); cf.append(int(keep.sum())+max(0,n_all-len(peaks)) if n_all else 0)   # kept(cap)+beyond-cap
    mp_all.append(ca); mp_filt.append(cf)
    if (k+1)%20==0: print(f'  {k+1}/{len(ib)}  (last field all {ca[-1]} filt {cf[-1]})',flush=True)
np.save('/root/lod_filter_mpall.npy',np.array(mp_all,dtype=object),allow_pickle=True)
np.save('/root/lod_filter_mpfilt.npy',np.array(mp_filt,dtype=object),allow_pickle=True)
pos=pcnt>0; neg=~pos
def acc(per,K): return np.array([sum(v[:K]) for v in per],float)
def wbacc(K): return np.array([sum(list(per_wb[i])[:K]) for i in range(len(per_wb))],float)
cf_wb=0.3613
bands=[('<1k',0,1000),('1k-5k',1000,5000),('5k-100k',5000,1e5),('>100k',1e5,1e12)]
out={'RP':RP,'cf_wb':cf_wb,'astar':ASTAR,'bstar':BSTAR,'CAP':CAP,'n_pos':int(pos.sum()),'n_neg':int(neg.sum()),'K_sweep':{}}
for K in KS:
    WB=wbacc(K); out['K_sweep'][K]={}
    for tag,MPper,re_mp,pr_mp in (('unfilt',mp_all,RP['re_all'],RP['pr_all']),('filt',mp_filt,RP['re_filt'],RP['pr_filt'])):
        cf_mp=pr_mp/max(re_mp,1e-6); MP=acc(MPper,K)
        pp=8000.0*(MP*cf_mp)/np.maximum(WB*cf_wb,1e-6)
        f=pp[neg]; mean_f=float(f.mean()); std_f=float(f.std()); thr=mean_f+Z*std_f
        LoD=Z*std_f/max(re_mp,1e-6)
        bs={}
        for nm,lo,hi in bands:
            m=pos&(ppul>=lo)&(ppul<hi); bs[nm]=[int(m.sum()),(round(float((pp[m]>thr).mean()),3) if m.sum() else None)]
        pm2=pos&(ppul>0)&(pp>0); rc=float(pearsonr(np.log10(pp[pm2]),np.log10(ppul[pm2]))[0]) if pm2.sum()>2 else None
        out['K_sweep'][K][tag]=dict(re_mp=round(re_mp,4),pr_mp=round(pr_mp,4),cf_mp=round(cf_mp,4),
            mean_f=round(mean_f,1),std_f=round(std_f,1),thr=round(thr,1),LoD=round(LoD,1),
            band_sens=bs,r_para=(round(rc,3) if rc else None))
        print(f'K={K:3d} {tag:7s}: re {re_mp:.3f} pr {pr_mp:.3f} | mean_f {mean_f:8.1f} std_f {std_f:8.1f} LoD {LoD:9.1f} MP/µL | r {rc}',flush=True)
        for nm,_,_ in bands: print(f'      band {nm:8s} n={bs[nm][0]:3d} sens@95spec={bs[nm][1]}',flush=True)
json.dump(out,open(RES/'results_filter.json','w'),indent=2); print('\nWROTE',RES/'results_filter.json'); print('LOD_FILTER_DONE',flush=True)
