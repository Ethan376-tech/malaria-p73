# -*- coding: utf-8 -*-
"""E2 (Tanzania external stain-invariance) + statistical hardening of the in-domain bag claim.
E2-primary : within-domain-C 5-fold linear probe (image pos/neg) per encoder.
E2-bonus   : B->C transfer (parasite/bg scorer trained on Chittagong-2, applied to Tanzania, max-tile).
Stats      : Ibadan in-domain repeated 5x3 CV (max-pool) from cached embeddings + paired ThickDINO vs RedDino.
Output -> outputs/experiments/e2_tanzania/"""
import os, sys, glob, random, json
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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

dev = 'cuda'; HALF = 112
norm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
RES = OUTPUTS/'experiments'/'e2_tanzania'; RES.mkdir(parents=True, exist_ok=True)
random.seed(0); np.random.seed(0)

# ---------- encoders ----------
from dinov2.models.vision_transformer import vit_small
def dv(sd, patch):
    m = vit_small(patch_size=patch, img_size=224, block_chunks=0, init_values=1.0)
    m.load_state_dict(sd, strict=False); return m.eval().to(dev)
red_sd = timm.create_model('hf_hub:Snarcy/RedDino-small', pretrained=True).state_dict()
encoders = {
    'SupViT':   timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0).eval().to(dev),
    'RedDino':  dv(red_sd, 14),
    'ThickDINO':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth', map_location='cpu'), 16),
    'SupCon':   dv(torch.load(CHECKPOINTS/'ssl_vits_supcon'/'encoder_final.pth', map_location='cpu'), 14),
}
print('encoders:', list(encoders), flush=True)

def encode(model, tiles_u8, bs=128):
    out = []
    for i in range(0, len(tiles_u8), bs):
        batch = torch.stack([norm(Image.fromarray(t)) for t in tiles_u8[i:i+bs]]).to(dev)
        with torch.no_grad(): out.append(model(batch).float().cpu().numpy())
    return np.concatenate(out)

# ---------- content-tile samplers ----------
def content_tiles(img, n, thr=18, tries=40):
    W, H = img.size; ts = []
    for _ in range(n*tries):
        cx = random.randint(HALF, max(HALF, W-HALF)); cy = random.randint(HALF, max(HALF, H-HALF))
        t = img.crop((cx-HALF, cy-HALF, cx+HALF, cy+HALF))
        if np.asarray(t.convert('L')).std() > thr:
            ts.append(np.asarray(t.resize((224,224))).astype(np.uint8))
            if len(ts) >= n: break
    while len(ts) < n: ts.append(np.zeros((224,224,3), np.uint8))
    return ts

# ---------- build Tanzania tile set ONCE ----------
N_PER = 6
inf  = sorted(glob.glob(str(RAW/'tanzania'/'Thick_Infected'/'*.jpg')))
unin = sorted(glob.glob(str(RAW/'tanzania'/'Thick_Uninfected'/'*.jpg')))
items = [(p,1) for p in inf] + [(p,0) for p in unin]
random.shuffle(items)
tz_tiles, tz_imgid, tz_lab = [], [], []
for k,(p,lab) in enumerate(items):
    try: im = Image.open(p).convert('RGB')
    except Exception: continue
    for t in content_tiles(im, N_PER):
        tz_tiles.append(t); tz_imgid.append(k); tz_lab.append(lab)
tz_tiles = np.stack(tz_tiles); tz_imgid = np.array(tz_imgid); tz_lab = np.array(tz_lab)
img_label = {k: lab for k,(p,lab) in enumerate(items)}
uniq_imgs = sorted(set(tz_imgid)); y_img = np.array([img_label[k] for k in uniq_imgs])
print(f'Tanzania: {len(items)} images ({sum(l for _,l in items)} pos), {len(tz_tiles)} tiles', flush=True)

# ---------- Chittagong-2 parasite/bg tiles (for B->C scorer) ----------
def chitt2_boxes(txt):
    cents = []
    for ln in txt.read_text(errors='ignore').splitlines()[1:]:
        f = ln.split(',')
        if len(f) >= 9 and f[1].strip() == 'Parasite':
            try: x1,y1,x2,y2 = map(float, f[5:9]); cents.append(((x1+x2)/2,(y1+y2)/2))
            except ValueError: pass
    return cents
def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)
def build_chitt2(n_tf=25):
    GT = CHITTAGONG2/'GT_updated'
    tfs = sorted([d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated']); random.shuffle(tfs)
    tiles, lab = [], []
    for tf in tfs[:n_tf]:
        for ip in sorted(tf.glob('*.jpg'))[:6]:
            bf = GT/tf.name/(ip.stem+'.txt')
            if not bf.exists(): continue
            cents = chitt2_boxes(bf)
            if len(cents) < 2: continue
            im = Image.open(ip).convert('RGB'); W,H = im.size
            for cx,cy in random.sample(cents, min(4,len(cents))):
                cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF))
                tiles.append(np.asarray(im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF)).resize((224,224))).astype(np.uint8)); lab.append(1)
            g=0
            for _ in range(40):
                rx,ry = random.randint(HALF,W-HALF), random.randint(HALF,H-HALF)
                if free(rx,ry,cents):
                    tiles.append(np.asarray(im.crop((rx-HALF,ry-HALF,rx+HALF,ry+HALF)).resize((224,224))).astype(np.uint8)); lab.append(0); g+=1
                if g>=4: break
    return np.stack(tiles), np.array(lab)
C2_tiles, C2_lab = build_chitt2()
print('Chittagong-2 scorer tiles:', len(C2_tiles), '| pos', int(C2_lab.sum()), flush=True)

# ============ E2 per encoder ============
rows_withinC, rows_b2c = [], []
for name, model in encoders.items():
    E_tz = encode(model, tz_tiles)                       # (Ntiles,384)
    # within-C: per-image MEAN-pool embedding
    img_emb = np.stack([E_tz[tz_imgid==k].mean(0) for k in uniq_imgs])
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr,te in skf.split(img_emb, y_img):
        sc = StandardScaler().fit(img_emb[tr]); lr = LogisticRegression(max_iter=3000).fit(sc.transform(img_emb[tr]), y_img[tr])
        aucs.append(roc_auc_score(y_img[te], lr.predict_proba(sc.transform(img_emb[te]))[:,1]))
    rows_withinC.append(dict(encoder=name, withinC_mean=round(np.mean(aucs),3), withinC_std=round(np.std(aucs),3)))
    # B->C transfer: train parasite/bg on Chittagong-2, score Tanzania tiles, per-image MAX
    E_c2 = encode(model, C2_tiles)
    sc = StandardScaler().fit(E_c2); lr = LogisticRegression(max_iter=3000).fit(sc.transform(E_c2), C2_lab)
    tile_scores = lr.predict_proba(sc.transform(E_tz))[:,1]
    img_max = np.array([tile_scores[tz_imgid==k].max() for k in uniq_imgs])
    rows_b2c.append(dict(encoder=name, B2C_maxtile_AUC=round(roc_auc_score(y_img, img_max),3)))
    print(f'{name}: within-C {rows_withinC[-1]["withinC_mean"]}+/-{rows_withinC[-1]["withinC_std"]} | B->C {rows_b2c[-1]["B2C_maxtile_AUC"]}', flush=True)

df_wc = pd.DataFrame(rows_withinC).set_index('encoder'); df_wc.to_csv(RES/'within_c.csv')
df_b2c = pd.DataFrame(rows_b2c).set_index('encoder'); df_b2c.to_csv(RES/'b2c_transfer.csv')

# ============ Stats hardening: Ibadan in-domain repeated 5x3 CV (cached emb, max-pool) ============
S = OUTPUTS/'experiments'/'stageD_bag'
bm = pd.read_csv(S/'bag_meta.csv'); ibmask = (bm.dataset=='ibadan').values
yb = bm.label01.values[ibmask]
embs = {n: np.load(S/f'emb_{n.lower() if n!="ThickDINO" else "thickdino"}.npy')[ibmask].max(1) for n in ['SupViT','RedDino','ThickDINO','SupCon']}
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)
fold_auc = {n: [] for n in embs}; splits = list(rskf.split(embs['ThickDINO'], yb))
for tr,te in splits:
    for n,X in embs.items():
        sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), yb[tr])
        fold_auc[n].append(roc_auc_score(yb[te], lr.predict_proba(sc.transform(X[te]))[:,1]))
def boot_ci(v, n=2000):
    v=np.array(v); idx=np.random.RandomState(0).randint(0,len(v),(n,len(v))); m=v[idx].mean(1); return np.percentile(m,[2.5,97.5])
rows_stat=[]
for n,v in fold_auc.items():
    ci = boot_ci(v); rows_stat.append(dict(encoder=n, ibadan_indomain_mean=round(np.mean(v),3), std=round(np.std(v),3),
                                           ci95_lo=round(ci[0],3), ci95_hi=round(ci[1],3)))
diff = np.array(fold_auc['ThickDINO']) - np.array(fold_auc['RedDino'])
paired = dict(mean_diff=round(diff.mean(),3), std_diff=round(diff.std(),3),
              pct_thickdino_better=round(100*(diff>0).mean(),1), dci=[round(x,3) for x in boot_ci(diff)])
df_stat = pd.DataFrame(rows_stat).set_index('encoder'); df_stat.to_csv(RES/'stats_indomain_repeatedCV.csv')
json.dump(paired, open(RES/'paired_thickdino_vs_reddino.json','w'), indent=2)
print('\n=== Ibadan in-domain repeated 5x3 CV ==='); print(df_stat.to_string())
print('paired ThickDINO-RedDino:', paired, flush=True)

# ============ figure ============
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':11})
fig, ax = plt.subplots(figsize=(7.2,4.0)); x=np.arange(len(df_wc)); w=0.38
ax.bar(x-w/2, df_wc['withinC_mean'], w, yerr=df_wc['withinC_std'], capsize=4, color='#4477aa', label='within-domain C (5-fold)')
ax.bar(x+w/2, df_b2c['B2C_maxtile_AUC'], w, color='#ee6677', label='B->C transfer (max-tile)')
ax.axhline(0.5, ls=':', color='gray'); ax.set_xticks(x); ax.set_xticklabels(df_wc.index)
ax.set_ylim(0.4,1.0); ax.set_ylabel('Tanzania image pos/neg ROC-AUC'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(RES/'fig_e2_tanzania.png', dpi=150, bbox_inches='tight')
print('\nWROTE ->', RES, flush=True)
