# -*- coding: utf-8 -*-
"""v2 parasite-aware MIL (Table 3.2 cross-site B->A + Table 3.3 top-k aggregations) on the 441-bag set.
Tile scorer = LogisticRegression on Chittagong-1 box tiles (parasite vs bg), per encoder (frozen features).
Score each Ibadan bag's 256 cached v2 tiles -> P(parasite); aggregate (max / top5/10/30 mean / mean).
Cross-site B->A = test Ibadan (clean). Averaged over the 3 tile-sampling seeds of the Ibadan bag embeddings.
Outputs -> stageD_bag/mil_parasite_aware_v2.csv, mil_aggregation_ablation_v2.csv"""
import os, sys, random, glob
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
random.seed(0); np.random.seed(0); dev='cuda'; HALF=112
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
red_sd=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()
encoders={'supvit':timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev),
          'reddino':dv(red_sd,14),
          'thickdino':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16),
          'supcon':dv(torch.load(CHECKPOINTS/'ssl_vits_supcon'/'encoder_final.pth',map_location='cpu'),14)}
D=OUTPUTS/'experiments'/'stageD_bag'; meta=pd.read_csv(D/'bag_meta_v2.csv')
SEEDF={0:'emb_{e}_v2.npy',1:'emb_{e}_v2_s1.npy',2:'emb_{e}_v2_s2.npy'}
EMB={s:{e:np.load(D/SEEDF[s].format(e=e)) for e in encoders} for s in SEEDF}
y=meta.label01.values; dom=meta.dataset.values; band=meta.para_band.values; A=(dom=='ibadan')
print('bags',len(meta),'| ibadan',int(A.sum()),'| seeds',list(SEEDF),flush=True)

# ---- Chittagong-1 box tiles (parasite vs bg) -> scorer per encoder ----
s=pd.read_csv(SPLITS/'splits.csv'); c1=s[s.dataset=='chittagong-1']
def boxes(txt):
    cs=[]
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f=ln.split(',')
        if len(f)>=9 and 'Parasit' in f[1]:
            try: x1,y1,x2,y2=map(float,f[5:9]); cs.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return cs
def cropc(im,cx,cy):
    W,H=im.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
tiles=[]; labs=[]
for r in c1.itertuples():
    img_dir=RAW/r.rel_path; ann_dir=str(img_dir).replace('All_PvTk','All_annotations')
    imgs=sorted(glob.glob(str(img_dir)+'/*.jpg')); random.shuffle(imgs)
    for ip in imgs[:6]:
        stem=os.path.splitext(os.path.basename(ip))[0]; af=os.path.join(ann_dir,stem+'.txt')
        cents=boxes(Path(af)) if os.path.exists(af) else []
        try: im=Image.open(ip).convert('RGB')
        except Exception: continue
        W,H=im.size
        for cx,cy in cents[:20]: tiles.append(np.asarray(cropc(im,cx,cy),np.uint8)); labs.append(1)
        g=0
        for _ in range(40):
            rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)
            if not cents or free(rx,ry,cents): tiles.append(np.asarray(cropc(im,rx,ry),np.uint8)); labs.append(0); g+=1
            if g>=max(3,len(cents[:20])): break
    if sum(labs)>=4000: break
labs=np.array(labs); print('scorer tiles',len(tiles),'parasite',int(labs.sum()),'bg',int((labs==0).sum()),flush=True)
@torch.no_grad()
def enc_tiles(model,bs=256):
    out=[]
    for i in range(0,len(tiles),bs):
        b=torch.stack([norm(Image.fromarray(t)) for t in tiles[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)
scorers={}
for e,m in encoders.items():
    X=enc_tiles(m); sc=StandardScaler().fit(X); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X),labs); scorers[e]=(sc,lr); print(e,'scorer trained',flush=True)

# ---- score Ibadan bags per seed, aggregate, average AUC over seeds ----
def Pmat(e,seed):
    sc,lr=scorers[e]; E=EMB[seed][e]; n=E.shape[0]
    return lr.predict_proba(sc.transform(E.reshape(-1,384)))[:,1].reshape(n,256)
def agg(P,mode):
    Ps=np.sort(P,1)[:,::-1]
    return {'max':Ps[:,0],'top5':Ps[:,:5].mean(1),'top10':Ps[:,:10].mean(1),'top30':Ps[:,:30].mean(1),'mean':P.mean(1)}[mode]
hi=A&((band=='neg')|(band=='pos_hi')); lo=A&((band=='neg')|(band=='pos_lo'))
MODES=['max','top5','top10','top30','mean']
pa_rows=[]; agg_rows=[]
for e in encoders:
    # per-seed AUCs
    auc_by_mode={mode:[] for mode in MODES}; auc_lo=[]; auc_hi=[]
    for seed in SEEDF:
        P=Pmat(e,seed)
        for mode in MODES: auc_by_mode[mode].append(roc_auc_score(y[A],agg(P,mode)[A]))
        auc_lo.append(roc_auc_score(y[lo],agg(P,'max')[lo])); auc_hi.append(roc_auc_score(y[hi],agg(P,'max')[hi]))
    bestmode=max(MODES,key=lambda m:np.mean(auc_by_mode[m]))
    pa_rows.append(dict(encoder=e, BA_maxP=round(np.mean(auc_by_mode['max']),3), BA_maxP_seedstd=round(np.std(auc_by_mode['max']),3),
                        BA_topk_best=round(np.mean(auc_by_mode[bestmode]),3), best_mode=bestmode,
                        lowpara_lo=round(np.mean(auc_lo),3), highpara_hi=round(np.mean(auc_hi),3)))
    agg_rows.append(dict(encoder=e, **{m:round(np.mean(auc_by_mode[m]),3) for m in MODES}))
    print(pa_rows[-1],flush=True)
pd.DataFrame(pa_rows).set_index('encoder').to_csv(D/'mil_parasite_aware_v2.csv')
pd.DataFrame(agg_rows).set_index('encoder').to_csv(D/'mil_aggregation_ablation_v2.csv')
print('\n=== parasite-aware v2 ==='); print(pd.DataFrame(pa_rows).to_string(index=False))
print('\n=== aggregation v2 ==='); print(pd.DataFrame(agg_rows).to_string(index=False))
print('XSITE_V2_DONE',flush=True)
