# -*- coding: utf-8 -*-
"""Robustness seeds for the v2 bag set: re-draw the 256 tiles/bag under 2 extra RNG seeds (1,2) and re-encode.
Bag order, labels, bands and e1_fold are IDENTICAL to bag_meta_v2.csv (only tile sampling changes), so rows align.
Loads the 4 encoders once, then encodes both seeds. Output -> emb_*_v2_s{1,2}.npy"""
import os, sys, random, json, time
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
from dinov2.models.vision_transformer import vit_small
dev='cuda'; K=256; TILE=224; MAX_FIELDS=12
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

# bag list (identical construction to build_bagset_v2.py, deterministic order independent of tile seed)
d=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
d=d[d.diagnosis.isin(['Pos','Neg'])].copy()
d['fdir']=[(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in d.iterrows()]
ib=[r.fdir for _,r in d.iterrows()]
s=pd.read_csv(SPLITS/'splits.csv'); ch=s[(s.dataset=='chittagong-1') & (s.label.isin(['pos','neg']))]
chb=[RAW/r.rel_path for _,r in ch.iterrows()]
fdirs=ib+chb
meta=pd.read_csv(OUTD/'bag_meta_v2.csv')
assert len(fdirs)==len(meta), f'align mismatch {len(fdirs)} vs {len(meta)}'
print(f'bags {len(fdirs)} | aligned to bag_meta_v2.csv', flush=True)

for SEED in [1,2]:
    random.seed(SEED); np.random.seed(SEED)
    emb={k:[] for k in encoders}; t0=time.time(); ok=0
    for i,fdir in enumerate(fdirs):
        tiles=sample_tiles(fdir)
        if not tiles:
            for k in encoders: emb[k].append(np.zeros((K,384),np.float32))   # keep alignment
            print('WARN no tiles row',i,flush=True); continue
        if len(tiles)<K: tiles=[tiles[j] for j in np.random.choice(len(tiles),K,replace=True)]
        for k,mdl in encoders.items(): emb[k].append(enc(mdl,tiles[:K]))
        ok+=1
        if (i+1)%80==0: print(f'seed{SEED} {i+1}/{len(fdirs)} | {(time.time()-t0)/60:.1f} min',flush=True)
    for k in encoders: np.save(OUTD/f'emb_{k}_v2_s{SEED}.npy', np.stack(emb[k]))
    print(f'SEED{SEED}_DONE ok={ok} in {(time.time()-t0)/60:.1f} min',flush=True)
print('EXTRASEEDS_DONE',flush=True)
