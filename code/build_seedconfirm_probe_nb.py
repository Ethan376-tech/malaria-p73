"""Generate experiments/notebooks/StageC_seedconfirm_probe.ipynb — multi-seed mean±std confirmation (EVAL §3.5)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Multi-seed confirmation (≥3 seeds, mean±std) — clean B→A tile probe (EVAL §3.5)

Re-runs the clean B→A tile probe over **5 seeds** (vary tile sampling + in-domain split) → mean±std AUC per
encoder, so headline claims are statistical (not single-run). Key arms: **SupViT, RedDino, ThickDINO (final,
pretrained-init+DINO adapt), ThickDINO-scratch v2, SupCon (final, box-sup)**. Saves table + bar-with-error-bars.""")

md("## 0 · Setup — load encoders once")
code(
"import os, sys, random, json\n"
"os.environ['XFORMERS_DISABLED']='1'\n"
"import numpy as np, torch, timm, pandas as pd\n"
"import torchvision.transforms as T\n"
"from PIL import Image; Image.MAX_IMAGE_PIXELS = None\n"
"sys.path.append('/root/A_Dissertation'); sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train')\n"
"from common.paths import CHITTAGONG2, RAW, CHECKPOINTS, OUTPUTS\n"
"dev='cuda'; norm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); HALF=112\n"
"from dinov2.models.vision_transformer import vit_small\n"
"def dv(sd,patch):\n"
"    m=vit_small(patch_size=patch, img_size=224, block_chunks=0, init_values=1.0); m.load_state_dict(sd,strict=False); return m.eval().to(dev)\n"
"red_sd=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()\n"
"encoders={'SupViT':timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).eval().to(dev),\n"
"          'RedDino':dv(red_sd,14),\n"
"          'ThickDINO':dv(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),16),\n"
"          'ThickDINO-scratch v2':dv(torch.load(CHECKPOINTS/'ssl_vits_thickdino_v2'/'encoder_final.pth',map_location='cpu'),14),\n"
"          'SupCon (box-sup)':dv(torch.load(CHECKPOINTS/'ssl_vits_supcon'/'encoder_final.pth',map_location='cpu'),14)}\n"
"RESDIR=OUTPUTS/'experiments'/'seedconfirm_probe'; RESDIR.mkdir(parents=True, exist_ok=True)\n"
"SEEDS=[0,1,2,3,4]\n"
"print('encoders:', list(encoders), '| seeds:', SEEDS)"
)

md("## 1 · Helpers (build tiles parameterised by seed)")
code(
"def crop(img,cx,cy):\n"
"    W,H=img.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return img.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))\n"
"def free(cx,cy,c): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in c)\n"
"def chitt_boxes(txt):\n"
"    cents=[]\n"
"    for ln in txt.read_text(errors='ignore').splitlines()[1:]:\n"
"        f=ln.split(',')\n"
"        if len(f)>=9 and f[1].strip()=='Parasite':\n"
"            try: x1,y1,x2,y2=map(float,f[5:9]); cents.append(((x1+x2)/2,(y1+y2)/2))\n"
"            except ValueError: pass\n"
"    return cents\n"
"def build_chitt(n_tf=25):\n"
"    GT=CHITTAGONG2/'GT_updated'; tfs=sorted([d for d in CHITTAGONG2.iterdir() if d.is_dir() and d.name!='GT_updated']); random.shuffle(tfs)\n"
"    tiles,lab,grp=[],[],[]\n"
"    for tf in tfs[:n_tf]:\n"
"        for ip in sorted(tf.glob('*.jpg'))[:6]:\n"
"            bf=GT/tf.name/(ip.stem+'.txt')\n"
"            if not bf.exists(): continue\n"
"            cents=chitt_boxes(bf)\n"
"            if len(cents)<2: continue\n"
"            im=Image.open(ip).convert('RGB'); W,H=im.size\n"
"            for cx,cy in random.sample(cents,min(4,len(cents))): tiles.append(crop(im,cx,cy)); lab.append(1); grp.append(tf.name)\n"
"            g=0\n"
"            for _ in range(40):\n"
"                rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)\n"
"                if free(rx,ry,cents): tiles.append(crop(im,rx,ry)); lab.append(0); grp.append(tf.name); g+=1\n"
"                if g>=4: break\n"
"    return tiles,np.array(lab),np.array(grp)\n"
"def build_fastmal(n_field=10):\n"
"    fm=RAW/'ibadan'/'FASTMAL_THICK_FILMS'; tiles,lab=[],[]\n"
"    for samp in sorted(p for p in fm.iterdir() if p.is_dir()):\n"
"        d=json.loads(list(samp.glob('*.json'))[0].read_text()); flds=d['rois']; random.shuffle(flds)\n"
"        for f in flds[:n_field]:\n"
"            ip=samp/f['image_name']\n"
"            if not ip.exists(): continue\n"
"            cents=[(float(r['x'])+float(r['width'])/2,float(r['y'])+float(r['height'])/2) for r in f['roi'] if 'PARASITE' in r.get('type','') and 'CROWD' not in r['type']]\n"
"            if len(cents)<1: continue\n"
"            im=Image.open(ip).convert('RGB'); W,H=im.size\n"
"            for cx,cy in random.sample(cents,min(4,len(cents))): tiles.append(crop(im,cx,cy)); lab.append(1)\n"
"            g=0\n"
"            for _ in range(40):\n"
"                rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)\n"
"                if free(rx,ry,cents): tiles.append(crop(im,rx,ry)); lab.append(0); g+=1\n"
"                if g>=4: break\n"
"    return tiles,np.array(lab)"
)

md("## 2 · Loop seeds × encoders → collect AUCs")
code(
"from sklearn.model_selection import GroupShuffleSplit\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.metrics import roc_auc_score\n"
"def encode(model,imgs,bs=128):\n"
"    out=[]\n"
"    for i in range(0,len(imgs),bs):\n"
"        b=torch.stack([norm(im) for im in imgs[i:i+bs]]).to(dev)\n"
"        with torch.no_grad(): out.append(model(b).float().cpu().numpy())\n"
"    return np.concatenate(out)\n"
"def fit(Etr,ytr,Ete,yte):\n"
"    sc=StandardScaler().fit(Etr); lr=LogisticRegression(max_iter=3000).fit(sc.transform(Etr),ytr)\n"
"    return roc_auc_score(yte,lr.predict_proba(sc.transform(Ete))[:,1])\n"
"rec={n:{'ind':[],'x':[]} for n in encoders}\n"
"for sd in SEEDS:\n"
"    random.seed(sd); np.random.seed(sd)\n"
"    Ct,Cy,Cg=build_chitt(); Ft,Fy=build_fastmal()\n"
"    tr,te=next(GroupShuffleSplit(1,test_size=0.3,random_state=sd).split(Ct,Cy,Cg))\n"
"    for n,m in encoders.items():\n"
"        Ec=encode(m,Ct); Ef=encode(m,Ft)\n"
"        rec[n]['ind'].append(fit(Ec[tr],Cy[tr],Ec[te],Cy[te])); rec[n]['x'].append(fit(Ec,Cy,Ef,Fy))\n"
"    print(f'seed {sd} done | C {len(Ct)} F {len(Ft)}')\n"
"rows=[]\n"
"for n in encoders:\n"
"    x=np.array(rec[n]['x']); ind=np.array(rec[n]['ind'])\n"
"    rows.append(dict(model=n, xdom_mean=round(x.mean(),3), xdom_std=round(x.std(),3),\n"
"                     indom_mean=round(ind.mean(),3), indom_std=round(ind.std(),3), n_seeds=len(SEEDS)))\n"
"res=pd.DataFrame(rows).set_index('model'); print(res.to_string())\n"
"try: display(res)\n"
"except: pass"
)

md("## 3 · Save + figure (mean±std, error bars)")
code(
"import matplotlib.pyplot as plt\n"
"res.to_csv(RESDIR/'seedconfirm_results.csv')\n"
"fig,ax=plt.subplots(figsize=(8,5))\n"
"ax.barh(res.index, res['xdom_mean'], xerr=res['xdom_std'], color='mediumseagreen', capsize=4)\n"
"ax.set_xlim(0.5,1.0); ax.set_xlabel('cross-domain lp-AUC (B->A), mean±std over %d seeds'%len(SEEDS)); ax.set_title('Encoder comparison (multi-seed)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'seedconfirm.png',dpi=150); plt.show()\n"
"print('saved ->', RESDIR)\n"
"td=res.loc['ThickDINO']; print(f'ThickDINO x-dom = {td.xdom_mean}±{td.xdom_std} | RedDino {res.loc[\"RedDino\",\"xdom_mean\"]} | SupViT {res.loc[\"SupViT\",\"xdom_mean\"]}')\n"
"print('ThickDINO beats RedDino (mean):', res.loc['ThickDINO','xdom_mean']>res.loc['RedDino','xdom_mean'])"
)

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
import os
out="/root/A_Dissertation/experiments/notebooks/StageC_seedconfirm_probe.ipynb"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote",out,"with",len(cells),"cells")
