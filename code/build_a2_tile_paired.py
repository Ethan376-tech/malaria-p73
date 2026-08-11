# -*- coding: utf-8 -*-
"""A2: paired significance test for the tile-level cross-site ranking (replaces 'non-overlapping intervals').
Lifts the exact seedconfirm probe pipeline, captures the 5 per-seed paired B->A AUCs for SupViT/RedDino/ThickDINO,
and runs paired tests (Wilcoxon signed-rank, paired t, sign) for ThickDINO vs RedDino and ThickDINO vs SupViT.
Output -> outputs/experiments/seedconfirm_probe/tile_paired_test.json"""
import os, sys, random, json
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, torch, timm, pandas as pd
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
sys.path.append('/root/A_Dissertation'); sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train')
from common.paths import CHITTAGONG2, RAW, CHECKPOINTS, OUTPUTS
from scipy.stats import wilcoxon, ttest_rel
dev='cuda'; norm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); HALF=112
from dinov2.models.vision_transformer import vit_small
def dv(sd,patch):
    m=vit_small(patch_size=patch, img_size=224, block_chunks=0, init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
red_sd=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()
encoders={'SupViT':timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev),
          'RedDino':dv(red_sd,14),
          'ThickDINO':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16)}
RESDIR=OUTPUTS/'experiments'/'seedconfirm_probe'; SEEDS=[0,1,2,3,4]

def crop(img,cx,cy):
    W,H=img.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return img.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
def chitt_boxes(txt):
    cents=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and f[1].strip()=='Parasite':
            try: x1,y1,x2,y2=map(float,f[5:9]); cents.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return cents
def build_chitt(n_tf=25):
    GT=CHITTAGONG2/'GT_updated'; tfs=sorted([d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated']); random.shuffle(tfs)
    tiles,lab,grp=[],[],[]
    for tf in tfs[:n_tf]:
        for ip in sorted(tf.glob('*.jpg'))[:6]:
            bf=GT/tf.name/(ip.stem+'.txt')
            if not bf.exists(): continue
            cents=chitt_boxes(bf)
            if len(cents)<2: continue
            im=Image.open(ip).convert('RGB'); W,H=im.size
            for cx,cy in random.sample(cents,min(4,len(cents))): tiles.append(crop(im,cx,cy)); lab.append(1); grp.append(tf.name)
            g=0
            for _ in range(40):
                rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
                if free(rx,ry,cents): tiles.append(crop(im,rx,ry)); lab.append(0); grp.append(tf.name); g+=1
                if g>=4: break
    return tiles,np.array(lab),np.array(grp)
def build_fastmal(n_field=10):
    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; tiles,lab=[],[]
    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):
        d=json.loads(list(samp.glob('*.json'))[0].read_text()); flds=d['rois']; random.shuffle(flds)
        for f in flds[:n_field]:
            ip=samp/f['image_name']
            if not ip.exists(): continue
            cents=[(float(r['x'])+float(r['width'])/2,float(r['y'])+float(r['height'])/2) for r in f['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r['type']]
            if len(cents)<1: continue
            im=Image.open(ip).convert('RGB'); W,H=im.size
            for cx,cy in random.sample(cents,min(4,len(cents))): tiles.append(crop(im,cx,cy)); lab.append(1)
            g=0
            for _ in range(40):
                rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
                if free(rx,ry,cents): tiles.append(crop(im,rx,ry)); lab.append(0); g+=1
                if g>=4: break
    return tiles,np.array(lab)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
def encode(model,imgs,bs=128):
    out=[]
    for i in range(0,len(imgs),bs):
        b=torch.stack([norm(im) for im in imgs[i:i+bs]]).to(dev)
        with torch.no_grad(): out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
def fit(Etr,ytr,Ete,yte):
    sc=StandardScaler().fit(Etr); lr=LogisticRegression(max_iter=3000).fit(sc.transform(Etr),ytr)
    return roc_auc_score(yte,lr.predict_proba(sc.transform(Ete))[:,1])

rec={n:[] for n in encoders}
for sd in SEEDS:
    random.seed(sd); np.random.seed(sd)
    Ct,Cy,Cg=build_chitt(); Ft,Fy=build_fastmal()
    for n,m in encoders.items():
        Ec=encode(m,Ct); Ef=encode(m,Ft); rec[n].append(round(float(fit(Ec,Cy,Ef,Fy)),4))
    print(f'seed {sd}: '+' '.join(f'{n}={rec[n][-1]}' for n in encoders), flush=True)

def paired(a,b):
    a=np.array(a); b=np.array(b); d=a-b
    out=dict(a_mean=round(float(a.mean()),4), b_mean=round(float(b.mean()),4),
             mean_diff=round(float(d.mean()),4), std_diff=round(float(d.std(ddof=1)),4),
             pooled_sd=round(float(np.sqrt((a.std(ddof=1)**2+b.std(ddof=1)**2)/2)),4),
             n_wins=int((d>0).sum()), n=len(d))
    try: out['paired_t_p']=round(float(ttest_rel(a,b).pvalue),5)
    except Exception as e: out['paired_t_p']=str(e)
    try: out['wilcoxon_p']=round(float(wilcoxon(a,b).pvalue),5)
    except Exception as e: out['wilcoxon_p']=str(e)
    # diff in units of pooled SD, and Cohen's dz (paired)
    out['diff_in_pooled_sd']=round(float(d.mean()/out['pooled_sd']),3) if out['pooled_sd']>0 else None
    out['cohens_dz']=round(float(d.mean()/d.std(ddof=1)),3) if d.std(ddof=1)>0 else None
    return out

report=dict(per_seed=rec,
            ThickDINO_vs_RedDino=paired(rec['ThickDINO'], rec['RedDino']),
            ThickDINO_vs_SupViT=paired(rec['ThickDINO'], rec['SupViT']),
            RedDino_vs_SupViT=paired(rec['RedDino'], rec['SupViT']))
json.dump(report, open(RESDIR/'tile_paired_test.json','w'), indent=2)
print(json.dumps(report, indent=2))
print('\nWROTE_DONE', RESDIR/'tile_paired_test.json', flush=True)
