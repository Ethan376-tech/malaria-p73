# -*- coding: utf-8 -*-
"""Robustify the B->C transfer test: 16 tiles/image, top-k(3) mean pooling (vs noisy max),
and 5 seeds varying the Chittagong-2 parasite/bg scorer training subset. Reports mean±std per encoder.
Tanzania tiles are encoded once per encoder (the bottleneck); seeds vary the B-trained scorer.
Output -> outputs/experiments/e2_tanzania/b2c_robust.csv + fig_b2c_robust.png"""
import os, sys, glob, random
os.environ['XFORMERS_DISABLED'] = '1'
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
sys.path.append('/root/A_Dissertation')
sys.path.insert(0, '/root/autodl-tmp/A_Dissertation/references/RedDino/train')
from common.paths import RAW, CHITTAGONG2, CHECKPOINTS, OUTPUTS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

dev='cuda'; HALF=112; N_PER=16; TOPK=3; SEEDS=[0,1,2,3,4]
norm=T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES=OUTPUTS/'experiments'/'e2_tanzania'; RES.mkdir(parents=True, exist_ok=True)

from dinov2.models.vision_transformer import vit_small
def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
red_sd=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()
encoders={'SupViT':timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev),
          'RedDino':dv(red_sd,14),
          'ThickDINO':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16),
          'SupCon':dv(torch.load(CHECKPOINTS/'ssl_vits_supcon'/'encoder_final.pth',map_location='cpu'),14)}
print('encoders:',list(encoders),flush=True)

def encode(model, tiles_u8, bs=128):
    out=[]
    for i in range(0,len(tiles_u8),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in tiles_u8[i:i+bs]]).to(dev)
        with torch.no_grad(): out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
def content_tiles(img,n,thr=18,tries=40):
    W,H=img.size; ts=[]
    for _ in range(n*tries):
        cx=random.randint(HALF,max(HALF,W-HALF)); cy=random.randint(HALF,max(HALF,H-HALF))
        t=img.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
        if np.asarray(t.convert('L')).std()>thr:
            ts.append(np.asarray(t.resize((224,224))).astype(np.uint8))
            if len(ts)>=n: break
    while len(ts)<n: ts.append(np.zeros((224,224,3),np.uint8))
    return ts

# Tanzania tiles ONCE (seed 0, 16/img)
random.seed(0); np.random.seed(0)
inf=sorted(glob.glob(str(RAW/'tanzania'/'Thick_Infected'/'*.jpg'))); unin=sorted(glob.glob(str(RAW/'tanzania'/'Thick_Uninfected'/'*.jpg')))
items=[(p,1) for p in inf]+[(p,0) for p in unin]; random.shuffle(items)
tz_tiles,tz_imgid,labmap=[],[],{}
for k,(p,lab) in enumerate(items):
    try: im=Image.open(p).convert('RGB')
    except Exception: continue
    for t in content_tiles(im,N_PER): tz_tiles.append(t); tz_imgid.append(k)
    labmap[k]=lab
tz_tiles=np.stack(tz_tiles); tz_imgid=np.array(tz_imgid)
uniq=sorted(set(tz_imgid)); y_img=np.array([labmap[k] for k in uniq])
print(f'Tanzania {len(items)} imgs, {len(tz_tiles)} tiles ({N_PER}/img)',flush=True)

def chitt2_boxes(txt):
    c=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and f[1].strip()=='Parasite':
            try: x1,y1,x2,y2=map(float,f[5:9]); c.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return c
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
def build_chitt2(seed,n_tf=25):
    rnd=random.Random(seed); GT=CHITTAGONG2/'GT_updated'
    tfs=sorted([d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated']); rnd.shuffle(tfs)
    tiles,lab=[],[]
    for tf in tfs[:n_tf]:
        for ip in sorted(tf.glob('*.jpg'))[:6]:
            bf=GT/tf.name/(ip.stem+'.txt')
            if not bf.exists(): continue
            cents=chitt2_boxes(bf)
            if len(cents)<2: continue
            im=Image.open(ip).convert('RGB'); W,H=im.size
            for cx,cy in rnd.sample(cents,min(4,len(cents))):
                cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF))
                tiles.append(np.asarray(im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF)).resize((224,224))).astype(np.uint8)); lab.append(1)
            g=0
            for _ in range(40):
                rx,ry=rnd.randint(HALF,W-HALF),rnd.randint(HALF,H-HALF)
                if free(rx,ry,cents): tiles.append(np.asarray(im.crop((rx-HALF,ry-HALF,rx+HALF,ry+HALF)).resize((224,224))).astype(np.uint8)); lab.append(0); g+=1
                if g>=4: break
    return np.stack(tiles), np.array(lab)

def topk_mean(scores, k=TOPK):
    return np.array([np.sort(scores[tz_imgid==i])[-k:].mean() for i in uniq])

rows=[]
for name,model in encoders.items():
    E_tz=encode(model,tz_tiles)                  # once
    auc_topk, auc_max = [], []
    for sd in SEEDS:
        C2t,C2l=build_chitt2(sd)
        E_c2=encode(model,C2t)
        sc=StandardScaler().fit(E_c2); lr=LogisticRegression(max_iter=3000).fit(sc.transform(E_c2),C2l)
        s=lr.predict_proba(sc.transform(E_tz))[:,1]
        auc_topk.append(roc_auc_score(y_img, topk_mean(s)))
        auc_max.append(roc_auc_score(y_img, np.array([s[tz_imgid==i].max() for i in uniq])))
    rows.append(dict(encoder=name,
                     B2C_topk_mean=round(np.mean(auc_topk),3), B2C_topk_std=round(np.std(auc_topk),3),
                     B2C_max_mean=round(np.mean(auc_max),3), B2C_max_std=round(np.std(auc_max),3)))
    print(f'{name}: top-{TOPK} {rows[-1]["B2C_topk_mean"]}+/-{rows[-1]["B2C_topk_std"]} | max {rows[-1]["B2C_max_mean"]}+/-{rows[-1]["B2C_max_std"]}',flush=True)

df=pd.DataFrame(rows).set_index('encoder'); df.to_csv(RES/'b2c_robust.csv')
print('\n',df.to_string(),flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':11})
fig,ax=plt.subplots(figsize=(7.2,4.0)); x=np.arange(len(df))
ax.bar(x, df['B2C_topk_mean'], 0.5, yerr=df['B2C_topk_std'], capsize=4, color='#ee6677')
ax.axhline(0.5, ls=':', color='gray'); ax.set_xticks(x); ax.set_xticklabels(df.index)
ax.set_ylim(0.3,1.0); ax.set_ylabel(f'B->C transfer AUC (top-{TOPK} mean, 5 seeds)'); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(RES/'fig_b2c_robust.png',dpi=150,bbox_inches='tight')
print('WROTE ->',RES,flush=True)
