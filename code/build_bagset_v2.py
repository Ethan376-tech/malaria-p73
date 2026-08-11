# -*- coding: utf-8 -*-
"""Rebuild the full MIL bag set with Petru's updated 242-sample Ibadan labels (was 209).
241 Ibadan (146 neg / 95 pos; 1 Null dropped) + 200 Chittagong-1 = 441 bags. New parasitaemia bands from
MP/WBC*8000 (<1000/uL = pos_lo), patient-grouped stratified 3-fold (e1_fold). Encode K=256 tiles with the 4
encoders. Output -> outputs/experiments/stageD_bag/emb_*_v2.npy + bag_meta_v2.csv"""
import os, sys, random, json, time
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
from dinov2.models.vision_transformer import vit_small
from sklearn.model_selection import StratifiedGroupKFold
random.seed(0); np.random.seed(0); dev='cuda'
K=256; TILE=224; MAX_FIELDS=12
stats=json.loads((SPLITS/'ssl_color_stats.json').read_text()); BLANK_THR=stats['blank_grey_std_threshold']
norm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
IMG_EXTS={'.jpg','.jpeg','.png','.tif','.tiff'}; OUTD=OUTPUTS/'experiments'/'stageD_bag'

def dv(sd,patch):
    m=vit_small(patch_size=patch,img_size=224,block_chunks=0,init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)
red_sd=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()
encoders={'supvit':timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev),
          'reddino':dv(red_sd,14),
          'thickdino':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16),
          'supcon':dv(torch.load(CHECKPOINTS/'ssl_vits_supcon'/'encoder_final.pth',map_location='cpu'),14)}
print('encoders ready', flush=True)

def gray_std(t): return float(np.asarray(t.convert('L'),np.float32).std())
def sample_tiles(fdir):
    fs=[p for p in fdir.rglob('*') if p.suffix.lower() in IMG_EXTS]; random.shuffle(fs); tiles=[]
    for f in fs[:MAX_FIELDS]:
        try: im=Image.open(f).convert('RGB')
        except Exception: continue
        W,H=im.size
        if min(W,H)<TILE: continue
        for _ in range(80):
            x=random.randint(0,W-TILE); y=random.randint(0,H-TILE); t=im.crop((x,y,x+TILE,y+TILE))
            if gray_std(t)>BLANK_THR: tiles.append(t)
            if len(tiles)>=K: break
        if len(tiles)>=K: break
    if 0<len(tiles)<K: tiles=[tiles[i] for i in np.random.choice(len(tiles),K,replace=True)]
    return tiles
@torch.no_grad()
def enc(model,tiles,bs=128):
    out=[]
    for i in range(0,len(tiles),bs):
        b=torch.stack([norm(t) for t in tiles[i:i+bs]]).to(dev); out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)

# ---- Ibadan from updated CSV ----
d=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
d=d[d.diagnosis.isin(['Pos','Neg'])].copy()
def dens(s):
    try: mp,wbc=str(s).split('/'); return float(mp)/float(wbc)*8000 if float(wbc)>0 else np.nan
    except Exception: return np.nan
d['dens']=d.parasite_count_MP_per_WBC.map(dens)
d['fdir']=[(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in d.iterrows()]
d['label01']=(d.diagnosis=='Pos').astype(int)
d['para_band']=['neg' if l==0 else ('pos_lo' if (dn<1000 or np.isnan(dn)) else 'pos_hi') for l,dn in zip(d.label01,d.dens)]
d['base']=d.sample_code.str.replace(r'r\d+$','',regex=True)
d['e1_fold']=-1
for f,(_,te) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=121).split(d,d.para_band,d.base)):
    d.iloc[te, d.columns.get_loc('e1_fold')]=f
ib=[dict(sample_id=r.sample_code,dataset='ibadan',domain='A_Nigeria',label01=int(r.label01),para_band=r.para_band,e1_fold=int(r.e1_fold),fdir=r.fdir) for _,r in d.iterrows()]
# ---- Chittagong-1 from splits.csv ----
s=pd.read_csv(SPLITS/'splits.csv'); ch=s[(s.dataset=='chittagong-1') & (s.label.isin(['pos','neg']))]
chb=[dict(sample_id=r.sample_id,dataset='chittagong-1',domain='B_Bangladesh',label01=1 if r.label=='pos' else 0,para_band=r.parasitaemia_band,e1_fold=int(r.e1_fold),fdir=RAW/r.rel_path) for _,r in ch.iterrows()]
bags=ib+chb
print(f'bags: {len(bags)} | ibadan {len(ib)} (pos {sum(b["label01"] for b in ib)}) | chitt {len(chb)}', flush=True)

emb={k:[] for k in encoders}; meta=[]; t0=time.time()
for r in bags:
    tiles=sample_tiles(r['fdir'])
    if not tiles: print('WARN no tiles', r['sample_id'], flush=True); continue
    for k,mdl in encoders.items(): emb[k].append(enc(mdl,tiles))
    meta.append(dict(sample_id=r['sample_id'],dataset=r['dataset'],domain=r['domain'],label01=r['label01'],para_band=r['para_band'],e1_fold=r['e1_fold'],n_tiles=len(tiles)))
    if len(meta)%40==0: print(f'{len(meta)}/{len(bags)} | {(time.time()-t0)/60:.1f} min', flush=True)
for k in encoders: np.save(OUTD/f'emb_{k}_v2.npy', np.stack(emb[k]))
pd.DataFrame(meta).to_csv(OUTD/'bag_meta_v2.csv', index=False)
print('DONE_V2', len(meta), 'bags in %.1f min'%((time.time()-t0)/60), '| pos_lo:', sum(1 for m in meta if m['para_band']=='pos_lo'), flush=True)
