# -*- coding: utf-8 -*-
"""S2.1 — ViT-native ICAM parameter calibration (replaces MILCA's VGG+GAP constants; §7 revised).
Per domain (A=FASTMAL/Ibadan, B=Chittagong-1), on a DEV split (test GT untouched), derive:
  (i)  GT parasite box-size distribution -> box_size + peak min-distance (px, our tile coord frame);
  (ii) CLS-similarity value distribution at GT-parasite patches vs background -> binarisation threshold (Youden on dev).
Frozen ThickDINO, CLS-sim = cos(patch, CLS). Output -> outputs/experiments/s2_calib/calib_constants.json (+ figure)."""
import os, sys, json, random
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import RAW, CHITTAGONG1, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.metrics import roc_auc_score, roc_curve
dev='cuda'; TILE=224; P=16; G=TILE//P    # 14x14 grid, 16px patches
random.seed(0); np.random.seed(0)
RES=OUTPUTS/'experiments'/'s2_calib'; RES.mkdir(parents=True, exist_ok=True)
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
m=vit_small(patch_size=P,img_size=224,block_chunks=0,init_values=1.0)
m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),strict=False)
m.eval().to(dev)
for p in m.parameters(): p.requires_grad_(False)

@torch.no_grad()
def clssim(tiles_u8, bs=32):
    out=[]
    for i in range(0,len(tiles_u8),bs):
        xb=torch.stack([norm(Image.fromarray(t)) for t in tiles_u8[i:i+bs]]).to(dev)
        ff=m.forward_features(xb); cls=ff['x_norm_clstoken']; patch=ff['x_norm_patchtokens']
        out.append(F.cosine_similarity(patch, cls.unsqueeze(1), dim=-1).cpu().numpy())   # (b,196)
    return np.concatenate(out)

def patch_labels(cents, x0, y0):
    lab=np.zeros(G*G,bool)
    for (bx,by) in cents:
        px,py=bx-x0,by-y0
        if 0<=px<TILE and 0<=py<TILE:
            gx,gy=int(px//P),int(py//P)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    nx,ny=gx+dx,gy+dy
                    if 0<=nx<G and 0<=ny<G: lab[ny*G+nx]=True
    return lab

# ---- domain A: FASTMAL (Ibadan) ----
def fastmal_fields():
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; out=[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        js=list(samp.glob('*.json'))
        if not js: continue
        for fld in json.loads(js[0].read_text())['rois']:
            ip=samp/fld['image_name']
            if not ip.exists(): continue
            boxes=[]
            for r in fld['roi']:
                t=r.get('type','')
                if 'PARASITE' in t and 'CROWD' not in t:
                    boxes.append((float(r['x']),float(r['y']),float(r['width']),float(r['height'])))
            if boxes: out.append((ip, boxes))
    return out

# ---- domain B: Chittagong-1 ----
def chitt_fields():
    ANN=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_annotations'; IMG=CHITTAGONG1/'NIH-NLM-ThickBloodSmearsPV'/'All_PvTk'
    out=[]
    for pv in sorted(d for d in ANN.iterdir() if d.is_dir()):
        for txt in sorted(pv.glob('*.txt')):
            boxes=[]
            for ln in txt.read_text(errors='ignore').splitlines()[1:]:
                f=ln.split(',')
                if len(f)>=9 and f[1].strip()=='Parasitized':
                    try:
                        x1,y1,x2,y2=map(float,f[5:9]); boxes.append((x1,y1,x2-x1,y2-y1))
                    except ValueError: pass
            ip=IMG/pv.name/(txt.stem+'.jpg')
            if boxes and ip.exists(): out.append((ip, boxes))
    return out

def calibrate(name, fields, max_dev_tiles=400):
    # dev/test split: even-indexed fields = DEV (test held out for S2.2)
    fields=sorted(fields, key=lambda z: str(z[0])); random.Random(0).shuffle(fields)
    dev_fields=fields[0::2]
    # (i) box sizes (px) on dev
    sides=[]
    for ip,boxes in dev_fields:
        for (x,y,w,h) in boxes: sides.append((w+h)/2.0)
    sides=np.array(sides); box_size=float(np.median(sides))
    # (ii) CLS-sim parasite vs bg on dev (parasite-centred tiles)
    tiles=[]; labs=[]
    for ip,boxes in dev_fields:
        cents=[(x+w/2,y+h/2) for (x,y,w,h) in boxes]
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        W,H=im.size
        if min(W,H)<TILE: continue
        for (cx,cy) in random.sample(cents,min(2,len(cents))):
            x0=int(min(max(cx-TILE/2,0),W-TILE)); y0=int(min(max(cy-TILE/2,0),H-TILE))
            lab=patch_labels(cents,x0,y0)
            if lab.any():
                tiles.append(np.asarray(im.crop((x0,y0,x0+TILE,y0+TILE)),np.uint8)); labs.append(lab)
        if len(tiles)>=max_dev_tiles: break
    S=clssim(tiles); y=np.concatenate(labs).astype(int); s=S.reshape(-1)
    auc=roc_auc_score(y,s); fpr,tpr,thr=roc_curve(y,s); j=np.argmax(tpr-fpr); sim_thr=float(thr[j])
    nms_dist=round(box_size*0.5)
    out=dict(domain=name, n_dev_fields=len(dev_fields), n_gt_boxes=int(len(sides)),
             box_size_px=round(box_size,1), box_size_p25=round(float(np.percentile(sides,25)),1),
             box_size_p75=round(float(np.percentile(sides,75)),1), peak_min_dist_px=nms_dist,
             sim_threshold=round(sim_thr,4), sim_sep_auc=round(auc,3), n_dev_tiles=len(tiles),
             parasite_patch_simmean=round(float(s[y==1].mean()),4), bg_patch_simmean=round(float(s[y==0].mean()),4))
    print(out, flush=True); return out, s, y

resA,sA,yA=calibrate('A_ibadan_fastmal', fastmal_fields())
resB,sB,yB=calibrate('B_chittagong1', chitt_fields())
calib={'A':resA,'B':resB,'note':'box_size=median GT side px (our tile coord frame); peak_min_dist=0.5*box_size; sim_threshold=Youden on dev CLS-sim parasite-vs-bg. Derived on DEV fields only (test held out).'}
json.dump(calib, open(RES/'calib_constants.json','w'), indent=2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':10})
fig,ax=plt.subplots(1,2,figsize=(10,4))
for k,(name,s,y,r) in enumerate([('A (Ibadan/FASTMAL)',sA,yA,resA),('B (Chittagong-1)',sB,yB,resB)]):
    ax[k].hist(s[y==0],bins=40,alpha=0.55,color='#4477aa',density=True,label='background patch')
    ax[k].hist(s[y==1],bins=40,alpha=0.55,color='#ee6677',density=True,label='parasite patch')
    ax[k].axvline(r['sim_threshold'],ls='--',color='black',lw=1.2,label=f"Youden thr {r['sim_threshold']:.2f}")
    ax[k].set_title(f"{name}\nbox {r['box_size_px']:.0f}px · sep-AUC {r['sim_sep_auc']:.2f}",fontsize=10)
    ax[k].set_xlabel('CLS-similarity'); ax[k].legend(fontsize=8)
ax[0].set_ylabel('density'); plt.tight_layout(); plt.savefig(RES/'fig_s2_calib.png',dpi=140,bbox_inches='tight')
print('S2_CALIB_DONE', flush=True)
