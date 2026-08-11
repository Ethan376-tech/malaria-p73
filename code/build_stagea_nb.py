"""Generate data_pipeline/03_ssl_patch_pool.ipynb (nbformat) — Stage A."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# 03 · Stage A — unlabelled SSL patch pool (for the DINOv2 encoder)

Builds the **leakage-safe field pool** the Stage-B DINOv2 trainer will crop 224×224 patches from on-the-fly
(no pre-saved patches → more augmentation diversity + less disk). Per DECISIONS / PREPROCESSING_AUG:

- **Pool = ibadan (A) + chittagong-1 (B) field images** (all, unlabelled — SSL uses no labels).
- **Excluded: tanzania (E2 external), chittagong-2 (E3 detection held-out), ibadan-FASTMAL (detection GT).**
- Each field tagged with dataset / domain / e1_fold / xdom_role → Stage B can carve inductive (per-direction)
  vs transductive SSL subsets. (Inductive-vs-transductive for E1 cross-domain = an OPEN choice, deferred to Stage B.)
- Also: **calibrate the blank-tile filter** (intensity-variance threshold) and **per-domain colour stats** (to
  calibrate the stain augmentation in Stage B).

Outputs: `data/splits/ssl_pool.csv` (field index) + `data/splits/ssl_color_stats.json`.""")

md("## 0 · Setup — SSL pool = ibadan + chittagong-1 fields (expand samples → field images)")
code(
"import sys, json, random\n"
"import numpy as np, pandas as pd\n"
"from pathlib import Path\n"
"from PIL import Image\n"
"Image.MAX_IMAGE_PIXELS = None\n"
"sys.path.append('/root/A_Dissertation')\n"
"from common.paths import RAW, SPLITS\n"
"random.seed(0); np.random.seed(0)\n"
"IMG_EXTS = {'.jpg','.jpeg','.png','.tif','.tiff'}\n"
"m = pd.read_csv(SPLITS / 'splits.csv')\n"
"pool_samples = m[m.dataset.isin(['ibadan','chittagong-1'])]      # excludes tanzania, chittagong-2, ibadan_fastmal\n"
"print('pool samples:', len(pool_samples), '| by dataset:', pool_samples.dataset.value_counts().to_dict())\n"
"rows = []\n"
"for r in pool_samples.itertuples():\n"
"    for f in (RAW / r.rel_path).rglob('*'):\n"
"        if f.suffix.lower() in IMG_EXTS:\n"
"            rows.append(dict(field_path=str(f.relative_to(RAW)).replace(chr(92),'/'),\n"
"                             dataset=r.dataset, domain=r.domain, sample_id=r.sample_id,\n"
"                             patient_group=r.patient_group, e1_fold=r.e1_fold,\n"
"                             xdom_role=r.xdom_role, label=r.label))\n"
"fields = pd.DataFrame(rows)\n"
"print('field images in pool:', len(fields))\n"
"print(fields.groupby('domain').size().to_string())"
)

md("## 1 · Calibrate blank-tile filter (intensity-variance threshold)\n\n"
   "Thick films have large empty background. Sample random 224 crops, measure greyscale std; a low-std crop is "
   "near-uniform background. We set a threshold to drop blanks at crop time in Stage B.")
code(
"TILE = 224\n"
"def rand_crop(im):\n"
"    W, H = im.size\n"
"    if min(W, H) < TILE: return None\n"
"    x = random.randint(0, W-TILE); y = random.randint(0, H-TILE)\n"
"    return im.crop((x, y, x+TILE, y+TILE))\n"
"def gray_std(tile):\n"
"    return float(np.asarray(tile.convert('L'), dtype=np.float32).std())\n"
"\n"
"stds, dom_of = [], []\n"
"for dom in ['A_Nigeria','B_Bangladesh']:\n"
"    sub = fields[fields.domain==dom].sample(min(60, (fields.domain==dom).sum()), random_state=0)\n"
"    for fp in sub.field_path:\n"
"        try: im = Image.open(RAW/fp).convert('RGB')\n"
"        except Exception: continue\n"
"        for _ in range(8):\n"
"            t = rand_crop(im)\n"
"            if t is not None: stds.append(gray_std(t)); dom_of.append(dom)\n"
"stds = np.array(stds); dom_of = np.array(dom_of)\n"
"THR = float(np.percentile(stds, 15))   # drop the most-uniform ~15% as blank\n"
"print(f'sampled {len(stds)} crops | grey-std: min {stds.min():.1f}, median {np.median(stds):.1f}, max {stds.max():.1f}')\n"
"print(f'BLANK threshold (15th pct grey-std) = {THR:.1f}  -> keep crops with std > {THR:.1f}')\n"
"import matplotlib.pyplot as plt\n"
"plt.figure(figsize=(7,3))\n"
"for dom in ['A_Nigeria','B_Bangladesh']: plt.hist(stds[dom_of==dom], bins=40, alpha=0.5, label=dom)\n"
"plt.axvline(THR, color='k', ls='--', label=f'blank thr {THR:.0f}'); plt.xlabel('grey std'); plt.legend(); plt.title('crop content (blank-tile filter)'); plt.show()"
)

md("## 2 · Per-domain colour statistics (to calibrate stain augmentation in Stage B)")
code(
"def content_patches(dom, n=200):\n"
"    out = []\n"
"    sub = fields[fields.domain==dom].sample(frac=1, random_state=1)\n"
"    for fp in sub.field_path:\n"
"        try: im = Image.open(RAW/fp).convert('RGB')\n"
"        except Exception: continue\n"
"        for _ in range(6):\n"
"            t = rand_crop(im)\n"
"            if t is not None and gray_std(t) > THR: out.append(np.asarray(t, dtype=np.float32)/255.)\n"
"            if len(out) >= n: return out\n"
"    return out\n"
"\n"
"stats = {}\n"
"for dom in ['A_Nigeria','B_Bangladesh']:\n"
"    P = np.stack(content_patches(dom))               # (n,224,224,3)\n"
"    stats[dom] = dict(rgb_mean=P.mean((0,1,2)).round(3).tolist(),\n"
"                      rgb_std=P.std((0,1,2)).round(3).tolist(), n=len(P))\n"
"sdf = pd.DataFrame({d: {**{f'mean_{c}': stats[d]['rgb_mean'][i] for i,c in enumerate('RGB')},\n"
"                        **{f'std_{c}':  stats[d]['rgb_std'][i]  for i,c in enumerate('RGB')}}\n"
"                    for d in stats}).T\n"
"print('per-domain RGB stats (content patches):'); print(sdf.to_string())\n"
"print('\\n=> A and B differ in colour/stain (motivates stain-aug ranges spanning A union B, pushed wider for C=tanzania).')"
)

md("## 3 · Save pool index + colour stats; sanity montage")
code(
"fields.to_csv(SPLITS/'ssl_pool.csv', index=False)\n"
"meta = dict(blank_grey_std_threshold=round(THR,2), tile=TILE,\n"
"            excluded=['tanzania (E2)','chittagong-2 (E3)','ibadan_fastmal'],\n"
"            per_domain_rgb=stats, n_fields=len(fields))\n"
"(SPLITS/'ssl_color_stats.json').write_text(json.dumps(meta, indent=2))\n"
"print('wrote', SPLITS/'ssl_pool.csv', '(', len(fields), 'fields ) and ssl_color_stats.json')\n"
"# montage: content crops A (top) vs B (bottom) — visualise the stain difference\n"
"fig, ax = plt.subplots(2, 5, figsize=(12,5))\n"
"for r_, dom in enumerate(['A_Nigeria','B_Bangladesh']):\n"
"    pts = content_patches(dom, n=5)\n"
"    for a, p in zip(ax[r_], pts): a.imshow(p); a.set_title(dom[:1]); a.axis('off')\n"
"plt.tight_layout(); plt.show()"
)

md("## 4 · Summary & self-check")
code(
"print('SSL field pool:', len(fields), 'field images')\n"
"print(fields.groupby(['domain','dataset']).size().to_string())\n"
"print('\\nest. effective patch dataset: ~', len(fields), 'fields x (many random 224 crops/field) -> >100k patches')\n"
"assert not fields.dataset.isin(['tanzania','chittagong-2','ibadan_fastmal']).any(), 'LEAKAGE: held-out data in SSL pool!'\n"
"assert fields.domain.isin(['A_Nigeria','B_Bangladesh']).all()\n"
"print('SELF-CHECK PASSED: pool excludes tanzania / chittagong-2 / FASTMAL; only domains A,B.')"
)

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python"}})
import os
out = "/root/A_Dissertation/data_pipeline/03_ssl_patch_pool.ipynb"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote", out, "with", len(cells), "cells")
