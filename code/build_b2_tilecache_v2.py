# -*- coding: utf-8 -*-
"""B2 v2 tile cache: 64 raw tiles/sample for the 441-bag v2 set (241 Ibadan + 200 Chittagong-1).
Labels / e1_fold / para_band taken from bag_meta_v2.csv (same as the in-domain re-run). rel_path: Ibadan from
Petru's CSV (input_file -> samples_part{1,2}/code), Chittagong-1 from splits.csv. CPU only (image sampling).
Output -> outputs/experiments/b2_tilecache_v2/ (per-sample .npy + manifest_v2.csv)"""
import sys, random, json, time
import numpy as np, pandas as pd
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.append('/root/A_Dissertation'); from common.paths import RAW, SPLITS, OUTPUTS, IBADAN_PART1, IBADAN_PART2
random.seed(0); np.random.seed(0)
K=64; TILE=224; MAX_FIELDS=12
stats=json.loads((SPLITS/'ssl_color_stats.json').read_text()); BLANK_THR=stats['blank_grey_std_threshold']
IMG_EXTS={'.jpg','.jpeg','.png','.tif','.tiff'}
CACHE=OUTPUTS/'experiments'/'b2_tilecache_v2'; CACHE.mkdir(parents=True, exist_ok=True)
D=OUTPUTS/'experiments'/'stageD_bag'; bm=pd.read_csv(D/'bag_meta_v2.csv')
# rel_path/fdir map
ibcsv=pd.read_csv('/root/autodl-tmp/A_Dissertation/data/raw/ibadan/sample_codes_parasite_diagnosis_crosscheck.csv')
ibdir={r.sample_code:(IBADAN_PART1 if 'part1' in r.input_file else IBADAN_PART2)/r.sample_code for _,r in ibcsv.iterrows()}
sp=pd.read_csv(SPLITS/'splits.csv'); chdir={r.sample_id:RAW/r.rel_path for _,r in sp[sp.dataset=='chittagong-1'].iterrows()}
def fdir_of(row):
    return ibdir[row.sample_id] if row.dataset=='ibadan' else chdir[row.sample_id]

def gray_std(t): return float(np.asarray(t.convert('L'),np.float32).std())
def sample_tiles(fdir):
    fs=[p for p in fdir.rglob('*') if p.suffix.lower() in IMG_EXTS]; random.shuffle(fs); tiles=[]
    for f in fs[:MAX_FIELDS]:
        try: im=Image.open(f).convert('RGB')
        except Exception: continue
        W,H=im.size
        if min(W,H)<TILE: continue
        for _ in range(120):
            x=random.randint(0,W-TILE); y=random.randint(0,H-TILE); t=im.crop((x,y,x+TILE,y+TILE))
            if gray_std(t)>BLANK_THR: tiles.append(np.asarray(t,np.uint8))
            if len(tiles)>=K: break
        if len(tiles)>=K: break
    if 0<len(tiles)<K:
        idx=np.random.choice(len(tiles),K,replace=True); tiles=[tiles[i] for i in idx]
    return tiles

meta=[]; t0=time.time()
for r in bm.itertuples():
    tiles=sample_tiles(fdir_of(r))
    if len(tiles)==0: print('WARN no tiles', r.sample_id, flush=True); continue
    np.save(CACHE/f'{r.sample_id}.npy', np.stack(tiles))
    meta.append(dict(sample_id=r.sample_id, dataset=r.dataset, domain=r.domain, label01=r.label01,
                     e1_fold=int(r.e1_fold), para_band=r.para_band, n_tiles=len(tiles)))
    if len(meta)%50==0: print(f'{len(meta)}/{len(bm)} | {(time.time()-t0)/60:.1f} min', flush=True)
pd.DataFrame(meta).to_csv(CACHE/'manifest_v2.csv', index=False)
print('B2CACHE_V2_DONE', len(meta), 'samples in %.1f min'%((time.time()-t0)/60), flush=True)
