"""Generate experiments/notebooks/StageB_supcon.ipynb — weakly-supervised contrastive encoder (chittagong-1 boxes)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Stage B (alt) · Weakly-supervised CONTRASTIVE encoder (SupCon) — task-aligned

Pure-DINO adaptation (v1-v3) couldn't beat frozen RedDino (near-saturated tile probe). **Fundamentally different
method:** use chittagong-1 (domain B) **parasite boxes** to train the encoder with **supervised contrastive
(SupCon)** on **parasite-tile vs background-tile**, + **strong stain aug spanning A∪B** for invariance. The
objective is now TASK-ALIGNED (parasite discrimination) and explicitly stain-invariant — directly the dissertation
theme, and matches the title's "contrastive". Init = RedDino (`init_values=1.0`, correct arch).

**Leakage:** trains ONLY on chittagong-1 (B). Validation probe trains on chittagong-2, tests FASTMAL-A — both
DISJOINT from chittagong-1 → clean. (For the final MIL E1 eval on chittagong-1 we'd restrict to train folds; here
the cross-domain probe is unaffected.) Output → `checkpoints/ssl_vits_supcon/` + `outputs/experiments/stageB_supcon/`.""")

md("## 0 · Setup & hyper-parameters")
code(
"import os, sys, math, random, time, json, glob\n"
"os.environ['XFORMERS_DISABLED']='1'\n"
"sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')\n"
"import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm\n"
"import torchvision.transforms as T, torchvision.transforms.functional as TF\n"
"from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n"
"from PIL import Image; Image.MAX_IMAGE_PIXELS=None\n"
"from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS\n"
"SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); dev='cuda'\n"
"N_STEPS=600; BATCH=128; SAVE_EVERY=150; TAU=0.1\n"
"LR_PROJ=1e-3; LR_BB=1e-4; WD=0.04; WARMUP=50; FREEZE=50\n"
"N_PAR_TARGET=6000; IMGS_PER_PATIENT=8; HALF=112\n"
"IMEAN=[0.485,0.456,0.406]; ISTD=[0.229,0.224,0.225]\n"
"OUTDIR=CHECKPOINTS/'ssl_vits_supcon'; OUTDIR.mkdir(parents=True,exist_ok=True)\n"
"RESDIR=OUTPUTS/'experiments'/'stageB_supcon'; RESDIR.mkdir(parents=True,exist_ok=True)\n"
"print('SupCon | steps',N_STEPS,'batch',BATCH,'lr proj/bb',LR_PROJ,LR_BB,'-> ',OUTDIR)"
)

md("## 1 · Extract chittagong-1 tiles — parasite (from boxes) vs background, in-memory, patient-grouped")
code(
"m=pd.read_csv(SPLITS/'splits.csv'); c1=m[m.dataset=='chittagong-1']\n"
"def boxes(txt):\n"
"    cs=[]\n"
"    for ln in txt.read_text(errors='ignore').splitlines()[1:]:\n"
"        f=ln.split(',')\n"
"        if len(f)>=9 and 'Parasit' in f[1]:\n"
"            try: x1,y1,x2,y2=map(float,f[5:9]); cs.append(((x1+x2)/2,(y1+y2)/2))\n"
"            except ValueError: pass\n"
"    return cs\n"
"def cropc(im,cx,cy):\n"
"    W,H=im.size; cx=int(min(max(cx,HALF),W-HALF)); cy=int(min(max(cy,HALF),H-HALF)); return im.crop((cx-HALF,cy-HALF,cx+HALF,cy+HALF))\n"
"def free(cx,cy,cents): return all(abs(bx-cx)>HALF or abs(by-cy)>HALF for bx,by in cents)\n"
"tiles=[]; labs=[]; grps=[]   # tiles as uint8 arrays\n"
"for r in c1.itertuples():\n"
"    img_dir=RAW/r.rel_path; ann_dir=str(img_dir).replace('All_PvTk','All_annotations')\n"
"    imgs=sorted(glob.glob(str(img_dir)+'/*.jpg')); random.shuffle(imgs)\n"
"    for ip in imgs[:IMGS_PER_PATIENT]:\n"
"        stem=os.path.splitext(os.path.basename(ip))[0]; af=os.path.join(ann_dir,stem+'.txt')\n"
"        cents=boxes(__import__('pathlib').Path(af)) if os.path.exists(af) else []\n"
"        try: im=Image.open(ip).convert('RGB')\n"
"        except Exception: continue\n"
"        W,H=im.size\n"
"        for cx,cy in cents[:30]:\n"
"            tiles.append(np.asarray(cropc(im,cx,cy),np.uint8)); labs.append(1); grps.append(r.patient_group)\n"
"        g=0\n"
"        for _ in range(60):\n"
"            rx,ry=random.randint(HALF,W-HALF),random.randint(HALF,H-HALF)\n"
"            if not cents or free(rx,ry,cents):\n"
"                tiles.append(np.asarray(cropc(im,rx,ry),np.uint8)); labs.append(0); grps.append(r.patient_group); g+=1\n"
"            if g>=max(3,len(cents[:30])): break\n"
"    if sum(labs)>=N_PAR_TARGET: break\n"
"labs=np.array(labs); grps=np.array(grps)\n"
"print('tiles:',len(tiles),'| parasite',int(labs.sum()),'| background',int((labs==0).sum()),'| patients',len(set(grps)))"
)

md("## 2 · Model — vit_small(init_values=1.0) from RedDino + projection head; SupCon loss; disc-LR")
code(
"from dinov2.models.vision_transformer import vit_small\n"
"red=timm.create_model('hf_hub:Snarcy/RedDino-small',pretrained=True).state_dict()\n"
"bb=vit_small(patch_size=14,img_size=224,block_chunks=0,init_values=1.0)\n"
"miss=bb.load_state_dict(red,strict=False); assert len(miss.unexpected_keys)==0, 'LayerScale dropped!'\n"
"print('backbone init RedDino | missing',len(miss.missing_keys),'unexpected',len(miss.unexpected_keys))\n"
"proj=nn.Sequential(nn.Linear(384,512),nn.GELU(),nn.Linear(512,128))\n"
"model=nn.ModuleDict({'bb':bb,'proj':proj}).to(dev)\n"
"for p in model['bb'].parameters(): p.requires_grad=False     # warmup proj first\n"
"opt=torch.optim.AdamW([{'params':model['proj'].parameters(),'lr':LR_PROJ},\n"
"                       {'params':model['bb'].parameters(),'lr':LR_BB}],weight_decay=WD)\n"
"BASE=[LR_PROJ,LR_BB]\n"
"def cosine(it,base,final,warm,total):\n"
"    if it<warm: return base*it/max(1,warm)\n"
"    p=(it-warm)/max(1,total-warm); return final+(base-final)*0.5*(1+math.cos(math.pi*p))\n"
"def supcon(z,y,tau=TAU):\n"
"    sim=z@z.t()/tau; N=z.size(0); eye=torch.eye(N,device=z.device,dtype=torch.bool)\n"
"    sim=sim.masked_fill(eye,-1e9); logp=sim-torch.logsumexp(sim,1,keepdim=True)\n"
"    pos=(y[:,None]==y[None,:])&~eye; pc=pos.sum(1).clamp(min=1)\n"
"    return (-(logp*pos).sum(1)/pc).mean()\n"
"print('SupCon model + loss ready')"
)

md("## 3 · Data loader (2 stain-augmented views) + train loop")
code(
"def gamma(img): return TF.adjust_gamma(img, random.uniform(0.7,1.4))\n"
"aug=T.Compose([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),\n"
"    T.RandomApply([T.RandomChoice([T.RandomRotation((90,90)),T.RandomRotation((180,180)),T.RandomRotation((270,270))])],p=0.6),\n"
"    T.ColorJitter(0.4,0.5,0.4,0.1), T.RandomApply([T.Lambda(gamma)],p=0.3),\n"
"    T.RandomApply([T.GaussianBlur(5,(0.1,2.0))],p=0.2), T.ToTensor(), T.Normalize(IMEAN,ISTD)])\n"
"class TileDS(Dataset):\n"
"    def __init__(self,tiles,labs): self.t=tiles; self.y=labs\n"
"    def __len__(self): return len(self.t)\n"
"    def __getitem__(self,i):\n"
"        im=Image.fromarray(self.t[i]); return aug(im),aug(im),int(self.y[i])\n"
"cnt=np.bincount(labs); w=np.where(labs==1,1.0/cnt[1],1.0/cnt[0])   # balance pos/neg\n"
"sampler=WeightedRandomSampler(w,num_samples=N_STEPS*BATCH,replacement=True)\n"
"loader=DataLoader(TileDS(tiles,labs),batch_size=BATCH,sampler=sampler,num_workers=10,pin_memory=True,drop_last=True,persistent_workers=True)\n"
"model.train(); t0=time.time(); losses=[]; step=0; unfrozen=False\n"
"for v1,v2,y in loader:\n"
"    v1,v2,y=v1.to(dev,non_blocking=True),v2.to(dev,non_blocking=True),y.to(dev)\n"
"    if (not unfrozen) and step>=FREEZE:\n"
"        for p in model['bb'].parameters(): p.requires_grad=True\n"
"        unfrozen=True; print(f'[step {step}] backbone UNFROZEN')\n"
"    for g,b in zip(opt.param_groups,BASE): g['lr']=cosine(step,b,b*0.01,WARMUP,N_STEPS)\n"
"    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
"        x=torch.cat([v1,v2]); yy=torch.cat([y,y])\n"
"        z=F.normalize(model['proj'](model['bb'](x).float()),dim=1)\n"
"        loss=supcon(z,yy)\n"
"    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.0); opt.step()\n"
"    losses.append(loss.item())\n"
"    if step%50==0: print(f'step {step:4d}/{N_STEPS} | supcon {np.mean(losses[-50:]):.3f} | bb_lr {opt.param_groups[1][\"lr\"]:.1e} | {(time.time()-t0)/max(1,step):.2f}s/step')\n"
"    if step and step%SAVE_EVERY==0: torch.save(model['bb'].state_dict(),OUTDIR/f'encoder_step{step}.pth')\n"
"    step+=1\n"
"    if step>=N_STEPS: break\n"
"print('done in %.1f min'%((time.time()-t0)/60))"
)

md("## 4 · Save encoder + artifacts")
code(
"torch.save(model['bb'].state_dict(),OUTDIR/'encoder_final.pth')\n"
"results=dict(method='SupCon weakly-supervised (chittagong-1 parasite-vs-bg)', steps=N_STEPS,batch=BATCH,tau=TAU,\n"
"             lr_proj=LR_PROJ,lr_bb=LR_BB,init='RedDino',arch='vit_small init_values=1.0',\n"
"             n_tiles=len(tiles),n_parasite=int(labs.sum()),final_loss=float(np.mean(losses[-50:])),min_loss=float(np.min(losses)))\n"
"for d in (OUTDIR,RESDIR): json.dump(results,open(d/'results.json','w'),indent=2)\n"
"np.save(RESDIR/'loss_history.npy',np.asarray(losses))\n"
"import matplotlib.pyplot as plt\n"
"plt.figure(figsize=(7,4)); plt.plot(np.convolve(losses,np.ones(20)/20,'valid')); plt.xlabel('step'); plt.ylabel('SupCon loss'); plt.title('Stage B SupCon (weakly-supervised)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'loss_curve.png',dpi=150); plt.show()\n"
"print('SAVED ->',OUTDIR,'|',RESDIR)\n"
"print('NEXT: probe SupCon checkpoints (init_values=1.0) vs RedDino 0.911 — same B->A protocol.')"
)

nb = new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
import os
out="/root/A_Dissertation/experiments/notebooks/StageB_supcon.ipynb"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote",out,"with",len(cells),"cells")
