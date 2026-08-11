"""Generate experiments/notebooks/StageB_dinov2_adapt_v3.ipynb — CORRECT arch (init_values=1.0) + gentle recipe."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Stage B v3 · CORRECT architecture (LayerScale) + gentle adaptation

v1/v2 were broken: `vit_small()` lacked LayerScale (`init_values=None`) → RedDino's 24 ls-gammas silently
dropped → frozen baseline scored 0.616 not 0.911. **v3 fix: `vit_small(..., init_values=1.0)`** (confirmed:
dinov2+RedDino → 0.911, matches timm). Same gentle recipe as v2: discriminative LR (head 1e-3 / backbone 1e-5),
backbone frozen 200 steps for head-warmup, grad-accum eff-batch 256, mild stain aug, 1000 steps. Now the
adaptation starts from a PROPER 0.911 encoder. Output → `checkpoints/ssl_vits_transductive_v3/` + `outputs/.../stageB_transductive_v3/`.""")

md("## 0 · Setup & hyper-parameters")
code(
"import os, sys, math, random, time, json\n"
"os.environ['XFORMERS_DISABLED'] = '1'\n"
"sys.path.insert(0, '/root/autodl-tmp/A_Dissertation/references/RedDino/train')\n"
"sys.path.append('/root/A_Dissertation')\n"
"import numpy as np, pandas as pd, torch, torch.nn as nn, timm\n"
"import torchvision.transforms as T, torchvision.transforms.functional as TF\n"
"from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n"
"from PIL import Image; Image.MAX_IMAGE_PIXELS = None\n"
"from common.paths import RAW, SPLITS, CHECKPOINTS, OUTPUTS\n"
"SEED=0; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n"
"dev='cuda'\n"
"N_STEPS=1000; BATCH=64; ACCUM=4; OUT_DIM=16384       # effective batch = 256\n"
"LR_HEAD=1e-3; LR_BB=1e-5; WD=0.04; WARMUP=50; FREEZE=200\n"
"S_TEMP=0.1; T_TEMP=(0.04,0.07,200); MOM=0.996\n"
"CROP=(180,300); SAVE_EVERY=200\n"
"IMEAN=[0.485,0.456,0.406]; ISTD=[0.229,0.224,0.225]\n"
"stats=json.loads((SPLITS/'ssl_color_stats.json').read_text()); BLANK_THR=stats['blank_grey_std_threshold']\n"
"OUTDIR=CHECKPOINTS/'ssl_vits_transductive_v3'; OUTDIR.mkdir(parents=True, exist_ok=True)\n"
"RESDIR=OUTPUTS/'experiments'/'stageB_transductive_v3'; RESDIR.mkdir(parents=True, exist_ok=True)\n"
"print('v3 (LayerScale fix) | steps',N_STEPS,'eff_batch',BATCH*ACCUM,'lr',LR_HEAD,LR_BB,'-> ',OUTDIR)"
)

md("## 1 · Data — 2 global views, mild stain aug, domain-balanced")
code(
"fields = pd.read_csv(SPLITS/'ssl_pool.csv')\n"
"def gamma(img): return TF.adjust_gamma(img, random.uniform(0.85,1.2))\n"
"photo = T.Compose([\n"
"    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),\n"
"    T.RandomApply([T.RandomChoice([T.RandomRotation((90,90)),T.RandomRotation((180,180)),T.RandomRotation((270,270))])],p=0.6),\n"
"    T.ColorJitter(0.2,0.3,0.2,0.05),\n"
"    T.RandomApply([T.Lambda(gamma)], p=0.2),\n"
"    T.RandomApply([T.GaussianBlur(5,(0.1,1.5))], p=0.1),\n"
"    T.ToTensor(), T.Normalize(IMEAN,ISTD)])\n"
"def scale_crop(im):\n"
"    W,H=im.size; t=None\n"
"    for _ in range(8):\n"
"        cs=min(random.randint(*CROP),W,H)\n"
"        x=random.randint(0,W-cs); y=random.randint(0,H-cs); t=im.crop((x,y,x+cs,y+cs))\n"
"        if np.asarray(t.convert('L'),dtype=np.float32).std()>BLANK_THR: break\n"
"    return t.resize((224,224))\n"
"class SSLPool(Dataset):\n"
"    def __init__(self, df): self.df=df.reset_index(drop=True)\n"
"    def __len__(self): return len(self.df)\n"
"    def __getitem__(self, i):\n"
"        for _ in range(3):\n"
"            try:\n"
"                im=Image.open(RAW/self.df.field_path[i]).convert('RGB')\n"
"                if min(im.size)>=64: return photo(scale_crop(im)), photo(scale_crop(im))\n"
"            except Exception: pass\n"
"            i=random.randint(0,len(self.df)-1)\n"
"        z=torch.zeros(3,224,224); return z,z\n"
"cnt=fields.domain.value_counts(); w=fields.domain.map(lambda d:1.0/cnt[d]).values\n"
"sampler=WeightedRandomSampler(w, num_samples=N_STEPS*ACCUM*BATCH, replacement=True)\n"
"loader=DataLoader(SSLPool(fields), batch_size=BATCH, sampler=sampler, num_workers=12,\n"
"                  pin_memory=True, drop_last=True, persistent_workers=True)\n"
"print('pool',len(fields),'fields | micro-batches=',N_STEPS*ACCUM)"
)

md("## 2 · Model — vit_small(init_values=1.0) [LAYERSCALE], init RedDino (clean), disc-LR, freeze backbone")
code(
"from dinov2.models.vision_transformer import vit_small\n"
"from dinov2.layers.dino_head import DINOHead\n"
"from dinov2.loss.dino_clstoken_loss import DINOLoss\n"
"red = timm.create_model('hf_hub:Snarcy/RedDino-small', pretrained=True).state_dict()\n"
"def backbone():\n"
"    m=vit_small(patch_size=14, img_size=224, block_chunks=0, init_values=1.0)   # <-- LayerScale FIX\n"
"    miss=m.load_state_dict(red, strict=False); return m, miss\n"
"sbb,miss=backbone(); tbb,_=backbone()\n"
"print('init from RedDino | missing',len(miss.missing_keys),'unexpected',len(miss.unexpected_keys),'(expect 1 / 0)')\n"
"assert len(miss.unexpected_keys)==0, 'LayerScale still dropped!'\n"
"shead=DINOHead(384,OUT_DIM); thead=DINOHead(384,OUT_DIM); thead.load_state_dict(shead.state_dict())\n"
"student=nn.ModuleDict({'bb':sbb,'head':shead}).to(dev)\n"
"teacher=nn.ModuleDict({'bb':tbb,'head':thead}).to(dev)\n"
"for p in teacher.parameters(): p.requires_grad=False\n"
"for p in student['bb'].parameters(): p.requires_grad=False     # head-warmup freeze\n"
"dino=DINOLoss(OUT_DIM).to(dev)\n"
"opt=torch.optim.AdamW([{'params':student['head'].parameters(),'lr':LR_HEAD},\n"
"                       {'params':student['bb'].parameters(),  'lr':LR_BB}], weight_decay=WD)\n"
"BASE=[LR_HEAD, LR_BB]\n"
"def cosine(it, base, final, warmup, total):\n"
"    if it<warmup: return base*it/max(1,warmup)\n"
"    p=(it-warmup)/max(1,total-warmup); return final+(base-final)*0.5*(1+math.cos(math.pi*p))\n"
"def ttemp(it):\n"
"    lo,hi,wn=T_TEMP; return lo+(hi-lo)*min(1.0,it/wn)\n"
"@torch.no_grad()\n"
"def ema(m):\n"
"    for ps,pt in zip(student.parameters(),teacher.parameters()): pt.mul_(m).add_(ps.detach(),alpha=1-m)\n"
"print('model OK (LayerScale present), optimizer ready')"
)

md("## 3 · Train loop — grad-accum, head-warmup then unfreeze")
code(
"student.train(); teacher.train(); t0=time.time(); losses=[]\n"
"step=0; micro=0; unfrozen=False; opt.zero_grad()\n"
"for v1,v2 in loader:\n"
"    v1,v2=v1.to(dev,non_blocking=True),v2.to(dev,non_blocking=True)\n"
"    if (not unfrozen) and step>=FREEZE:\n"
"        for p in student['bb'].parameters(): p.requires_grad=True\n"
"        unfrozen=True; print(f'[step {step}] backbone UNFROZEN (lr {LR_BB})')\n"
"    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
"        s1=student['head'](student['bb'](v1)); s2=student['head'](student['bb'](v2))\n"
"        with torch.no_grad():\n"
"            t1=teacher['head'](teacher['bb'](v1)); t2=teacher['head'](teacher['bb'](v2))\n"
"            tt=ttemp(step); t1c=dino.sinkhorn_knopp_teacher(t1,tt); t2c=dino.sinkhorn_knopp_teacher(t2,tt)\n"
"        loss=(dino([s1],[t2c])+dino([s2],[t1c]))/2\n"
"    (loss/ACCUM).backward(); micro+=1\n"
"    if micro%ACCUM==0:\n"
"        for g,b in zip(opt.param_groups,BASE): g['lr']=cosine(step,b,b*0.01,WARMUP,N_STEPS)\n"
"        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],3.0)\n"
"        opt.step(); opt.zero_grad(); ema(MOM); losses.append(loss.item())\n"
"        if step%50==0:\n"
"            print(f'step {step:4d}/{N_STEPS} | loss {np.mean(losses[-50:]):.3f} | head_lr {opt.param_groups[0][\"lr\"]:.1e}'\n"
"                  f' bb_lr {opt.param_groups[1][\"lr\"]:.1e} | {(time.time()-t0)/max(1,step):.2f}s/step')\n"
"        if step and step%SAVE_EVERY==0: torch.save(student['bb'].state_dict(), OUTDIR/f'encoder_step{step}.pth')\n"
"        step+=1\n"
"        if step>=N_STEPS: break\n"
"print('done in %.1f min'%((time.time()-t0)/60))"
)

md("## 4 · Save adapted encoder + artifacts")
code(
"torch.save(student['bb'].state_dict(), OUTDIR/'encoder_final.pth')\n"
"results=dict(steps=N_STEPS,eff_batch=BATCH*ACCUM,lr_head=LR_HEAD,lr_bb=LR_BB,freeze=FREEZE,mom=MOM,\n"
"             pool='transductive_A+B', init='RedDino', arch='vit_small init_values=1.0 (LayerScale)',\n"
"             recipe='v3: arch-fix + gentle (disc-LR, head-warmup, accum256, mild-aug)',\n"
"             final_loss=float(np.mean(losses[-50:])), loss_first=float(losses[0]), min_loss=float(np.min(losses)))\n"
"for d in (OUTDIR, RESDIR): json.dump(results, open(d/'results.json','w'), indent=2)\n"
"np.save(RESDIR/'loss_history.npy', np.asarray(losses))\n"
"pd.DataFrame({'step':range(len(losses)),'dino_loss':losses}).to_csv(RESDIR/'loss_history.csv', index=False)\n"
"import matplotlib.pyplot as plt\n"
"plt.figure(figsize=(7,4)); plt.plot(np.convolve(losses,np.ones(20)/20,'valid'))\n"
"plt.axvline(FREEZE,ls='--',c='g',label='unfreeze'); plt.legend()\n"
"plt.xlabel('opt step'); plt.ylabel('DINO loss (smooth-20)'); plt.title('Stage B v3 (LayerScale-correct)')\n"
"plt.tight_layout(); plt.savefig(RESDIR/'loss_curve.png', dpi=150); plt.show()\n"
"print('SAVED ->', OUTDIR, '|', RESDIR)\n"
"print('NEXT: probe v3 (init_values=1.0) vs frozen RedDino 0.911 — fair comparison at last.')"
)

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python"}})
import os
out = "/root/A_Dissertation/experiments/notebooks/StageB_dinov2_adapt_v3.ipynb"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,"w") as f: nbf.write(nb,f)
print("wrote", out, "with", len(cells), "cells")
