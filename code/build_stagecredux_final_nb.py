"""Generate experiments/notebooks/StageC_redux_final.ipynb — comprehensive §2.2 encoder comparison (clean B->A probe)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Stage C-redux FINAL · comprehensive §2.2 encoder comparison (clean B→A tile probe)

All encoders via `vit_small(init_values=1.0)`, same B→A protocol (train chittagong-2, test FASTMAL-A — both
EXCLUDED from ssl_pool/SupCon → CLEAN for every encoder). The full §2.2 story:
- **ImageNet** (A1, generic supervised) · **RedDino frozen** (A3, blood SSL foundation, =0.911 baseline)
- **SupCon final** (Path-1, box-supervised contrastive — beat RedDino on tile probe)
- **FromScratch step2000..final** (Path-2, OUR own DINO+iBOT from random init on thick-film tiles)
Saves table+figure → `outputs/experiments/stageC_redux_final/`.""")

md("## 0 · Setup — all encoders")
code(
"import os, sys, random, json\n"
"os.environ['XFORMERS_DISABLED']='1'\n"
"import numpy as np, torch, timm, pandas as pd\n"
"import torchvision.transforms as T\n"
"from PIL import Image; Image.MAX_IMAGE_PIXELS = None\n"
"sys.path.append('/root/A_Dissertation'); sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train')\n"
"from common.paths import CHITTAGONG2, RAW, CHECKPOINTS, OUTPUTS\n"
"random.seed(0); np.random.seed(0); dev='cuda'\n"
"norm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); HALF=112\n"
"from dinov2.models.vision_transformer import vit_small\n"
"red_sd = timm.create_model('hf_hub:Snarcy/RedDino-small', pretrained=True).state_dict()\n"
"def dv(sd):\n"
"    m=vit_small(patch_size=14, img_size=224, block_chunks=0, init_values=1.0); m.load_state_dict(sd, strict=False); return m.eval().to(dev)\n"
"imnet = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0).eval().to(dev)\n"
"FS=CHECKPOINTS/'ssl_vits_fromscratch'; SC=CHECKPOINTS/'ssl_vits_supcon'\n"
"encoders={'ImageNet ViT-S':imnet, 'RedDino (same-fwd)':dv(red_sd), 'SupCon final (box-sup)':dv(torch.load(SC/'encoder_final.pth',map_location='cpu'))}\n"
"for s in ['step2000','step4000','step6000','step8000','step10000']: encoders[f'FromScratch {s}']=dv(torch.load(FS/f'encoder_{s}.pth',map_location='cpu'))\n"
"encoders['FromScratch final']=dv(torch.load(FS/'encoder_final.pth',map_location='cpu'))\n"
"RESDIR=OUTPUTS/'experiments'/'stageC_redux_final'; RESDIR.mkdir(parents=True, exist_ok=True)\n"
"print('encoders:', list(encoders))"
)

md("## 1 · Helpers + build tiles (B->A, verbatim)")
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
"    return tiles,np.array(lab)\n"
"Ctiles,Cy,Cg = build_chitt(); Ftiles,Fy = build_fastmal()\n"
"print('chittagong-B',len(Ctiles),'| FASTMAL-A',len(Ftiles))"
)

md("## 2 · Encode + probe (in-domain B + cross-domain B->A)")
code(
"from sklearn.model_selection import GroupShuffleSplit\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.neighbors import KNeighborsClassifier\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.metrics import roc_auc_score\n"
"def encode(model,imgs,bs=128):\n"
"    out=[]\n"
"    for i in range(0,len(imgs),bs):\n"
"        b=torch.stack([norm(im) for im in imgs[i:i+bs]]).to(dev)\n"
"        with torch.no_grad(): out.append(model(b).float().cpu().numpy())\n"
"    return np.concatenate(out)\n"
"def fit_eval(Etr,ytr,Ete,yte):\n"
"    sc=StandardScaler().fit(Etr); Xtr,Xte=sc.transform(Etr),sc.transform(Ete)\n"
"    lr=LogisticRegression(max_iter=3000).fit(Xtr,ytr); lp=roc_auc_score(yte,lr.predict_proba(Xte)[:,1])\n"
"    kn=KNeighborsClassifier(20).fit(Xtr,ytr); kp=roc_auc_score(yte,kn.predict_proba(Xte)[:,1])\n"
"    return lp,kp\n"
"def row(name,model):\n"
"    Ec=encode(model,Ctiles); Ef=encode(model,Ftiles)\n"
"    tr,te=next(GroupShuffleSplit(1,test_size=0.3,random_state=0).split(Ec,Cy,Cg))\n"
"    ind_lp,_=fit_eval(Ec[tr],Cy[tr],Ec[te],Cy[te]); x_lp,x_kn=fit_eval(Ec,Cy,Ef,Fy)\n"
"    return dict(model=name, indomain_lpAUC=round(ind_lp,3), xdomain_lpAUC=round(x_lp,3), lp_drop=round(ind_lp-x_lp,3), xdomain_knnAUC=round(x_kn,3))\n"
"res=pd.DataFrame([row(n,m) for n,m in encoders.items()]).set_index('model')\n"
"print(res.to_string())\n"
"try: display(res)\n"
"except: pass"
)

md("## 3 · Save + verdict")
code(
"import matplotlib.pyplot as plt\n"
"res.to_csv(RESDIR/'crossdomain_probe_results.csv')\n"
"base=res.loc['RedDino (same-fwd)','xdomain_lpAUC']\n"
"fig,ax=plt.subplots(figsize=(8,6))\n"
"res['xdomain_lpAUC'].plot.barh(ax=ax,color='slateblue'); ax.axvline(base,ls='--',c='r',label=f'RedDino {base}'); ax.axvline(0.81,ls=':',c='gray',label='ImageNet 0.81'); ax.set_xlim(0.5,1.0); ax.legend(); ax.set_xlabel('cross-domain lp-AUC (B->A)'); ax.set_title('§2.2 encoder comparison (clean B->A)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'encoder_comparison.png',dpi=150); plt.show()\n"
"fs=res[res.index.str.startswith('FromScratch')]; bestfs=fs['xdomain_lpAUC'].idxmax()\n"
"print(f'RedDino baseline x-dom={base} | ImageNet=0.81 | SupCon final={res.loc[\"SupCon final (box-sup)\",\"xdomain_lpAUC\"]}')\n"
"print(f'BEST FromScratch (Path2): {bestfs} x-dom={fs.loc[bestfs,\"xdomain_lpAUC\"]}')\n"
"print('Path2 vs ImageNet:', 'ABOVE' if fs['xdomain_lpAUC'].max()>0.81 else 'BELOW', '| vs RedDino:', 'ABOVE' if fs['xdomain_lpAUC'].max()>base else 'BELOW (expected: from-scratch on 305k < foundation 1.25M)')"
)

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
import os
out="/root/A_Dissertation/experiments/notebooks/StageC_redux_final.ipynb"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote",out,"with",len(cells),"cells")
