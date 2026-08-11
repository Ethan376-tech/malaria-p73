"""Generate experiments/notebooks/MoCo_fair.ipynb — fair MoCo (queue + PREDICTOR head) for the framework ablation."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells=[]
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# MoCo-fair · queue MoCo + PREDICTOR head (fair framework ablation vs DINO+iBOT)

Strengthened MoCo so the DINO-vs-MoCo comparison is fair: add a **BYOL/MoCo-v3-style predictor** on the query
(breaks symmetry, the standard stabiliser the quick version lacked), keep the memory queue + symmetric InfoNCE,
and match EVERYTHING else to ThickDINO (SupViT-init patch16, same 0.8M tile data, same strong aug, 15k steps,
batch 96). Only the SSL objective differs (MoCo vs DINO+iBOT). Output → `checkpoints/ssl_vits_moco_fair/`.""")

md("## 0 · Setup")
code(
"import os, sys, math, random, time, json, glob\n"
"os.environ['XFORMERS_DISABLED']='1'\n"
"sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')\n"
"import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm\n"
"import torchvision.transforms as T, torchvision.transforms.functional as TF\n"
"from torch.utils.data import Dataset, DataLoader\n"
"from PIL import Image; Image.MAX_IMAGE_PIXELS=None\n"
"from common.paths import PROCESSED, CHECKPOINTS, OUTPUTS\n"
"SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); dev='cuda'\n"
"PATCH=16; N_STEPS=15000; BATCH=96; PROJ_DIM=256; QUEUE_K=96*100\n"
"BASE_LR=2e-4; WD=0.04; WARMUP=500; TAU=0.2; MOM=0.99; SAVE_EVERY=3000\n"
"IMEAN=[0.485,0.456,0.406]; ISTD=[0.229,0.224,0.225]\n"
"TILES=PROCESSED/'ssl_tiles_path2_big'\n"
"OUTDIR=CHECKPOINTS/'ssl_vits_moco_fair'; OUTDIR.mkdir(parents=True,exist_ok=True)\n"
"RESDIR=OUTPUTS/'experiments'/'moco_fair'; RESDIR.mkdir(parents=True,exist_ok=True)\n"
"print('MoCo-fair (+predictor) | steps',N_STEPS,'batch',BATCH,'queue',QUEUE_K,'-> ',OUTDIR)"
)

md("## 1 · Data — 2 strong-aug views (same as ThickDINO/MoCo)")
code(
"paths=sorted(glob.glob(str(TILES/'A'/'*.jpg'))+glob.glob(str(TILES/'B'/'*.jpg'))); print('tiles:',len(paths))\n"
"def gamma(img): return TF.adjust_gamma(img, random.uniform(0.7,1.4))\n"
"aug=T.Compose([T.RandomResizedCrop(224,scale=(0.32,1.0),antialias=True), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),\n"
"    T.RandomApply([T.RandomChoice([T.RandomRotation((90,90)),T.RandomRotation((180,180)),T.RandomRotation((270,270))])],p=0.6),\n"
"    T.ColorJitter(0.4,0.4,0.4,0.1), T.RandomApply([T.Lambda(gamma)],p=0.3), T.RandomApply([T.GaussianBlur(5,(0.1,2.0))],p=0.3),\n"
"    T.RandomSolarize(0.5,p=0.1), T.ToTensor(), T.Normalize(IMEAN,ISTD)])\n"
"class Tiles(Dataset):\n"
"    def __init__(self,p): self.p=p\n"
"    def __len__(self): return len(self.p)\n"
"    def __getitem__(self,i):\n"
"        try: im=Image.open(self.p[i]).convert('RGB')\n"
"        except Exception: im=Image.new('RGB',(224,224))\n"
"        return aug(im), aug(im)\n"
"loader=DataLoader(Tiles(paths),batch_size=BATCH,shuffle=True,num_workers=12,pin_memory=True,drop_last=True,persistent_workers=True)\n"
"print('batches/epoch',len(loader))"
)

md("## 2 · Model — query(bb+proj+PREDICTOR) / key(bb+proj EMA) + queue")
code(
"from dinov2.models.vision_transformer import vit_small\n"
"sup_sd=timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).state_dict()\n"
"def bbone():\n"
"    m=vit_small(patch_size=PATCH,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1); m.load_state_dict(sup_sd,strict=False); return m\n"
"def proj(): return nn.Sequential(nn.Linear(384,2048),nn.GELU(),nn.Linear(2048,PROJ_DIM))\n"
"def pred(): return nn.Sequential(nn.Linear(PROJ_DIM,512),nn.BatchNorm1d(512),nn.GELU(),nn.Linear(512,PROJ_DIM))  # query-only predictor\n"
"fq=nn.ModuleDict({'bb':bbone(),'proj':proj(),'pred':pred()}).to(dev)\n"
"fk=nn.ModuleDict({'bb':bbone(),'proj':proj()}).to(dev)\n"
"fk['bb'].load_state_dict(fq['bb'].state_dict()); fk['proj'].load_state_dict(fq['proj'].state_dict())\n"
"for p in fk.parameters(): p.requires_grad=False\n"
"queue=F.normalize(torch.randn(PROJ_DIM,QUEUE_K,device=dev),dim=0); qptr=[0]\n"
"opt=torch.optim.AdamW(fq.parameters(),lr=BASE_LR,weight_decay=WD)\n"
"def cos(it,a,b,warm,tot):\n"
"    if it<warm: return a*it/max(1,warm)\n"
"    p=(it-warm)/max(1,tot-warm); return b+(a-b)*0.5*(1+math.cos(math.pi*p))\n"
"qpar=lambda: list(fq['bb'].parameters())+list(fq['proj'].parameters())\n"
"kpar=lambda: list(fk['bb'].parameters())+list(fk['proj'].parameters())\n"
"@torch.no_grad()\n"
"def ema(m):\n"
"    for ps,pt in zip(qpar(),kpar()): pt.mul_(m).add_(ps.detach(),alpha=1-m)\n"
"def emb_q(x): return F.normalize(fq['pred'](fq['proj'](fq['bb'](x))),dim=1)\n"
"@torch.no_grad()\n"
"def emb_k(x): return F.normalize(fk['proj'](fk['bb'](x)),dim=1)\n"
"def infonce(q,k):\n"
"    pos=(q*k).sum(1,keepdim=True); neg=q@queue; logits=torch.cat([pos,neg],1)/TAU\n"
"    return F.cross_entropy(logits, torch.zeros(q.size(0),dtype=torch.long,device=dev))\n"
"@torch.no_grad()\n"
"def enqueue(k):\n"
"    B=k.size(0); p=qptr[0]; queue[:,p:p+B]=k.T; qptr[0]=(p+B)%QUEUE_K\n"
"print('MoCo-fair model ready (predictor added)')"
)

md("## 3 · Train loop")
code(
"fq.train(); fk.train(); t0=time.time(); L=[]; step=0; it=iter(loader)\n"
"while step<N_STEPS:\n"
"    try: v1,v2=next(it)\n"
"    except StopIteration: it=iter(loader); v1,v2=next(it)\n"
"    v1,v2=v1.to(dev,non_blocking=True),v2.to(dev,non_blocking=True)\n"
"    for g in opt.param_groups: g['lr']=cos(step,BASE_LR,1e-6,WARMUP,N_STEPS)\n"
"    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
"        q1=emb_q(v1); q2=emb_q(v2); k1=emb_k(v1); k2=emb_k(v2)\n"
"        loss=infonce(q1,k2)+infonce(q2,k1)\n"
"    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fq.parameters(),3.0); opt.step()\n"
"    ema(MOM); enqueue(k1.detach().float())\n"
"    L.append(loss.item())\n"
"    if step%100==0: print(f'step {step:5d}/{N_STEPS} | infonce {np.mean(L[-100:]):.3f} | {(time.time()-t0)/max(1,step):.2f}s/it')\n"
"    if step and step%SAVE_EVERY==0: torch.save(fq['bb'].state_dict(),OUTDIR/f'encoder_step{step}.pth')\n"
"    step+=1\n"
"print('done in %.1f min'%((time.time()-t0)/60))"
)

md("## 4 · Save")
code(
"torch.save(fq['bb'].state_dict(),OUTDIR/'encoder_final.pth')\n"
"json.dump(dict(method='MoCo-fair (queue+predictor), SupViT-init',steps=N_STEPS,batch=BATCH,queue=QUEUE_K,tau=TAU,patch=PATCH,init='SupViT',final_loss=float(np.mean(L[-100:]))),open(OUTDIR/'results.json','w'),indent=2)\n"
"np.save(RESDIR/'loss_history.npy',np.array(L))\n"
"import matplotlib.pyplot as plt\n"
"plt.figure(figsize=(8,4)); plt.plot(np.convolve(L,np.ones(50)/50,'valid')); plt.xlabel('step'); plt.ylabel('InfoNCE'); plt.title('MoCo-fair (+predictor)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'loss_curve.png',dpi=150); plt.show()\n"
"print('SAVED ->',OUTDIR)\n"
"print('NEXT: multi-seed framework probe — MoCo-fair vs DINO+iBOT vs RedDino vs SupViT.')"
)

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
import os
out="/root/A_Dissertation/experiments/notebooks/MoCo_fair.ipynb"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote",out,"with",len(cells),"cells")
