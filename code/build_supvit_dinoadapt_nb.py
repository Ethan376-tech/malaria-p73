"""Generate experiments/notebooks/SupViT_DINOadapt.ipynb — D-Q1(b): ImageNet-init + DINO+iBOT adapt, softmax centering."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# D-Q1(b) · SupViT-init + DINO+iBOT adaptation (softmax centering) — fair vs SupViT

Fair test: init the encoder from **SupViT (ImageNet ViT-S/16) weights** (same base as the SupViT baseline) via
`vit_small(patch_size=16, init_values=1.0)` (verified to reproduce SupViT's 0.810 at start), then **DINO+iBOT
adapt** on thick-film tiles. Contrast SupViT (frozen) vs this (ImageNet + our SSL) cleanly isolates "does our
self-supervised adaptation add value on top of supervised ImageNet?".

D-Q2 fixes baked in: **softmax/EMA centering instead of Sinkhorn-Knopp** (small-batch-robust — Sinkhorn needs
large batch, caused the v3 collapse), keep **local crops** (help here), **no KoLeo**, batch 96 / 8192 prototypes
(stable ratio), **gentle LR** (adapt a strong init without forgetting). Output → `checkpoints/ssl_vits_supvit_dinoadapt/`.""")

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
"PATCH=16; LOCAL_SZ=96; NP=196   # 224/16=14 ->196 patches ; local 96/16=6\n"
"N_STEPS=15000; BATCH=96; OUT_DIM=8192; N_LOCAL=6\n"
"BASE_LR=2e-4; WD=0.04; WD_END=0.2; WARMUP=500   # gentle (adapting a strong init)\n"
"S_TEMP=0.1; T_TEMP=(0.04,0.07,2000); MOM=(0.994,1.0)\n"
"P_MASK=0.5; MASK_R=(0.1,0.5); SAVE_EVERY=3000\n"
"IMEAN=[0.485,0.456,0.406]; ISTD=[0.229,0.224,0.225]\n"
"TILES=PROCESSED/'ssl_tiles_path2_big'\n"
"OUTDIR=CHECKPOINTS/'ssl_vits_supvit_dinoadapt'; OUTDIR.mkdir(parents=True,exist_ok=True)\n"
"RESDIR=OUTPUTS/'experiments'/'supvit_dinoadapt'; RESDIR.mkdir(parents=True,exist_ok=True)\n"
"print('SupViT-init DINO adapt | steps',N_STEPS,'batch',BATCH,'patch',PATCH,'softmax-centering ->',OUTDIR)"
)

md("## 1 · Data — 2 global (224) + 6 local (96), strong aug")
code(
"paths=sorted(glob.glob(str(TILES/'A'/'*.jpg'))+glob.glob(str(TILES/'B'/'*.jpg')))\n"
"print('tiles:',len(paths))\n"
"def gamma(img): return TF.adjust_gamma(img, random.uniform(0.7,1.4))\n"
"def photo(extra):\n"
"    return T.Compose(extra+[T.RandomHorizontalFlip(), T.ColorJitter(0.4,0.4,0.4,0.1),\n"
"        T.RandomApply([T.Lambda(gamma)],p=0.3), T.RandomApply([T.GaussianBlur(5,(0.1,2.0))],p=0.3), T.ToTensor(), T.Normalize(IMEAN,ISTD)])\n"
"g_aug=photo([T.RandomResizedCrop(224,scale=(0.32,1.0),antialias=True), T.RandomVerticalFlip(),\n"
"    T.RandomApply([T.RandomChoice([T.RandomRotation((90,90)),T.RandomRotation((180,180)),T.RandomRotation((270,270))])],p=0.6),\n"
"    T.RandomSolarize(0.5,p=0.1)])\n"
"l_aug=photo([T.RandomResizedCrop(LOCAL_SZ,scale=(0.05,0.32),antialias=True)])\n"
"class Tiles(Dataset):\n"
"    def __init__(self,p): self.p=p\n"
"    def __len__(self): return len(self.p)\n"
"    def __getitem__(self,i):\n"
"        try: im=Image.open(self.p[i]).convert('RGB')\n"
"        except Exception: im=Image.new('RGB',(224,224))\n"
"        return g_aug(im), g_aug(im), torch.stack([l_aug(im) for _ in range(N_LOCAL)])\n"
"loader=DataLoader(Tiles(paths), batch_size=BATCH, shuffle=True, num_workers=12, pin_memory=True, drop_last=True, persistent_workers=True)\n"
"print('batches/epoch', len(loader))"
)

md("## 2 · Model — vit_small(patch16, init_values=1.0) init from SupViT + DINOHead; softmax centering")
code(
"from dinov2.models.vision_transformer import vit_small\n"
"from dinov2.layers.dino_head import DINOHead\n"
"from dinov2.loss.dino_clstoken_loss import DINOLoss\n"
"sup_sd=timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=True,num_classes=0).state_dict()\n"
"def backbone():\n"
"    m=vit_small(patch_size=PATCH,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)\n"
"    miss=m.load_state_dict(sup_sd,strict=False); return m,miss\n"
"sbb,miss=backbone(); tbb,_=backbone()\n"
"print('SupViT init | missing',len(miss.missing_keys),'unexpected',len(miss.unexpected_keys),'(expect ~25: ls+mask, 0)')\n"
"assert len(miss.unexpected_keys)==0\n"
"head_s=DINOHead(384,OUT_DIM); head_t=DINOHead(384,OUT_DIM)\n"
"student=nn.ModuleDict({'bb':sbb,'head':head_s}).to(dev)\n"
"teacher=nn.ModuleDict({'bb':tbb,'head':head_t}).to(dev)\n"
"teacher.load_state_dict(student.state_dict())\n"
"for p in teacher.parameters(): p.requires_grad=False\n"
"dino=DINOLoss(OUT_DIM).to(dev); ibot_center=torch.zeros(1,1,OUT_DIM,device=dev)\n"
"opt=torch.optim.AdamW(student.parameters(), lr=BASE_LR, weight_decay=WD)\n"
"def cos(it,a,b,warm,tot):\n"
"    if it<warm: return a*it/max(1,warm)\n"
"    p=(it-warm)/max(1,tot-warm); return b+(a-b)*0.5*(1+math.cos(math.pi*p))\n"
"def cos_up(it,a,b,tot): p=min(1.0,it/tot); return a+(b-a)*0.5*(1-math.cos(math.pi*p))\n"
"def ttemp(it): lo,hi,wn=T_TEMP; return lo+(hi-lo)*min(1.0,it/wn)\n"
"@torch.no_grad()\n"
"def ema(m):\n"
"    for ps,pt in zip(student.parameters(),teacher.parameters()): pt.mul_(m).add_(ps.detach(),alpha=1-m)\n"
"def make_masks(B):\n"
"    m=torch.zeros(B,NP,dtype=torch.bool)\n"
"    for i in range(B):\n"
"        if random.random()<P_MASK: m[i,torch.randperm(NP)[:int(NP*random.uniform(*MASK_R))]]=True\n"
"    return m\n"
"print('softmax-centering DINO (not Sinkhorn) | no KoLeo | gentle adapt of SupViT')"
)

md("## 3 · Train loop — DINO multi-crop (softmax-centered teacher) + iBOT")
code(
"student.train(); teacher.train(); t0=time.time(); L=[];Ld=[];Li=[]; step=0; it=iter(loader)\n"
"while step<N_STEPS:\n"
"    try: g1,g2,loc=next(it)\n"
"    except StopIteration: it=iter(loader); g1,g2,loc=next(it)\n"
"    B=g1.size(0); g1,g2=g1.to(dev,non_blocking=True),g2.to(dev,non_blocking=True); loc=loc.to(dev,non_blocking=True).flatten(0,1)\n"
"    m1=make_masks(B).to(dev); m2=make_masks(B).to(dev)\n"
"    lr=cos(step,BASE_LR,1e-6,WARMUP,N_STEPS); wd=cos_up(step,WD,WD_END,N_STEPS)\n"
"    for g in opt.param_groups: g['lr']=lr; g['weight_decay']=wd\n"
"    tt=ttemp(step); mom=cos_up(step,MOM[0],MOM[1],N_STEPS)\n"
"    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
"        os1=student['bb'].forward_features(g1,m1); os2=student['bb'].forward_features(g2,m2); ol=student['bb'].forward_features(loc)\n"
"        with torch.no_grad():\n"
"            ot1=teacher['bb'].forward_features(g1); ot2=teacher['bb'].forward_features(g2)\n"
"            tg1=teacher['head'](ot1['x_norm_clstoken']); tg2=teacher['head'](ot2['x_norm_clstoken'])\n"
"            tg=torch.cat([tg1,tg2]); tcen=dino.softmax_center_teacher(tg,tt); tc1,tc2=tcen[:B],tcen[B:]   # SOFTMAX centering\n"
"        scg1=student['head'](os1['x_norm_clstoken']); scg2=student['head'](os2['x_norm_clstoken'])\n"
"        scl=student['head'](ol['x_norm_clstoken']).view(B,N_LOCAL,-1)\n"
"        terms=[dino([scg1],[tc2]), dino([scg2],[tc1])]\n"
"        for l in range(N_LOCAL): terms+=[dino([scl[:,l]],[tc1]), dino([scl[:,l]],[tc2])]\n"
"        loss_dino=sum(terms)/len(terms)\n"
"        def ibot_term(os_,ot_,m):\n"
"            sp=student['head'](os_['x_norm_patchtokens'])\n"
"            with torch.no_grad():\n"
"                tp=teacher['head'](ot_['x_norm_patchtokens']); tprob=F.softmax((tp-ibot_center)/tt,dim=-1)\n"
"            per=(tprob*F.log_softmax(sp/S_TEMP,dim=-1)).sum(-1); mf=m.float()\n"
"            return (-(per*mf).sum(-1)/mf.sum(-1).clamp(min=1.0)).mean(), tp.detach()\n"
"        li1,tp1=ibot_term(os1,ot1,m1); li2,tp2=ibot_term(os2,ot2,m2); loss_ibot=(li1+li2)/2\n"
"        loss=loss_dino+loss_ibot\n"
"    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),3.0); opt.step()\n"
"    ema(mom); dino.update_center(tg)   # EMA center update (softmax centering)\n"
"    with torch.no_grad(): ibot_center.mul_(0.9).add_(0.1*torch.cat([tp1,tp2]).mean(dim=(0,1),keepdim=True))\n"
"    L.append(loss.item()); Ld.append(loss_dino.item()); Li.append(loss_ibot.item())\n"
"    if step%100==0: print(f'step {step:5d}/{N_STEPS} | loss {np.mean(L[-100:]):.3f} (dino {np.mean(Ld[-100:]):.3f} ibot {np.mean(Li[-100:]):.3f}) | lr {lr:.1e} | {(time.time()-t0)/max(1,step):.2f}s/it')\n"
"    if step and step%SAVE_EVERY==0: torch.save(student['bb'].state_dict(),OUTDIR/f'encoder_step{step}.pth')\n"
"    step+=1\n"
"print('done in %.1f min'%((time.time()-t0)/60))"
)

md("## 4 · Save + loss figure")
code(
"torch.save(student['bb'].state_dict(),OUTDIR/'encoder_final.pth')\n"
"results=dict(method='D-Q1(b): SupViT-init + DINO+iBOT adapt, softmax centering',steps=N_STEPS,batch=BATCH,patch=PATCH,\n"
"             out_dim=OUT_DIM,base_lr=BASE_LR,init='SupViT(ImageNet)',centering='softmax',n_tiles=len(paths),\n"
"             final_loss=float(np.mean(L[-100:])),final_dino=float(np.mean(Ld[-100:])),final_ibot=float(np.mean(Li[-100:])))\n"
"for d in (OUTDIR,RESDIR): json.dump(results,open(d/'results.json','w'),indent=2)\n"
"np.savez(RESDIR/'loss_history.npz', total=np.array(L), dino=np.array(Ld), ibot=np.array(Li))\n"
"import matplotlib.pyplot as plt\n"
"plt.figure(figsize=(9,4))\n"
"for arr,lab in [(L,'total'),(Ld,'dino'),(Li,'ibot')]: plt.plot(np.convolve(arr,np.ones(50)/50,'valid'),label=lab)\n"
"plt.legend(); plt.xlabel('step'); plt.ylabel('loss'); plt.title('SupViT-init DINO+iBOT adapt (softmax centering)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'loss_curve.png',dpi=150); plt.show()\n"
"print('SAVED ->',OUTDIR,'|',RESDIR)\n"
"print('NEXT: probe vs SupViT 0.81 (start) — does DINO adaptation improve ImageNet features?')"
)

nb=new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
import os
out="/root/A_Dissertation/experiments/notebooks/SupViT_DINOadapt.ipynb"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote",out,"with",len(cells),"cells")
