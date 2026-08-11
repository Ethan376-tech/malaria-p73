# -*- coding: utf-8 -*-
"""In-domain MIL fine-tune of ThickDINO on Chittagong-1 (domain B) sample labels (purely weak), for the §2.3
generator/localiser on B. Mirrors the Ibadan FT in build_s2_2_v3.py. Saves enc+head. (Caveat: B annotated fields
overlap the bag samples -> localisation eval on B is in-domain-with-overlap; flagged in the write-up. A/FASTMAL is clean.)"""
import os, sys, time
os.environ['XFORMERS_DISABLED']='1'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/root/autodl-tmp/A_Dissertation/references/RedDino/train'); sys.path.append('/root/A_Dissertation')
from common.paths import CHECKPOINTS, OUTPUTS
from dinov2.models.vision_transformer import vit_small
dev='cuda'; np.random.seed(0); torch.manual_seed(0)
CACHE=OUTPUTS/'experiments'/'b2_tilecache_v2_128'
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
def td():
    m=vit_small(patch_size=16,img_size=224,block_chunks=0,init_values=1.0,drop_path_rate=0.1)
    m.load_state_dict(torch.load(CHECKPOINTS/'ssl_vits_supvit_dinoadapt'/'encoder_final.pth',map_location='cpu'),strict=False); return m
class MILNet(nn.Module):
    def __init__(self): super().__init__(); self.enc=td(); self.head=nn.Sequential(nn.Dropout(0.2),nn.Linear(384,1))
    def forward(self,x): B,K=x.shape[:2]; h=self.enc(x.flatten(0,1)).view(B,K,-1).amax(1); return self.head(h).squeeze(-1)
man=pd.read_csv(CACHE/'manifest_v2_128.csv'); ch=man[man.dataset=='chittagong-1'].reset_index(drop=True)
ids=ch.sample_id.tolist(); y=ch.set_index('sample_id').label01
def load_bag(sid,k):
    a=np.load(CACHE/f'{sid}.npy'); idx=np.random.choice(len(a),k,replace=len(a)<k); return a[idx]
def to_dev(arrs): x=torch.from_numpy(np.stack(arrs)).to(dev).float().div(255).permute(0,1,4,2,3); return (x-MEAN.unsqueeze(1))/STD.unsqueeze(1)
net=MILNet().to(dev)
opt=torch.optim.AdamW([{'params':net.enc.parameters(),'lr':5e-5},{'params':net.head.parameters(),'lr':1e-3}],weight_decay=0.05)
ya=np.array([y[s] for s in ids]); pw=torch.tensor([(ya==0).sum()/max(1,(ya==1).sum())],device=dev,dtype=torch.float32)
print(f'chittagong bags {len(ids)} pos {int(ya.sum())}',flush=True); t0=time.time()
for ep in range(6):
    net.train(); order=np.random.permutation(len(ids))
    for i in range(0,len(ids),2):
        bids=[ids[j] for j in order[i:i+2]]
        x=to_dev([load_bag(s,32) for s in bids]); yb=torch.tensor([y[s] for s in bids],dtype=torch.float32,device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=F.binary_cross_entropy_with_logits(net(x),yb,pos_weight=pw)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),3.0); opt.step()
    print(f'ep{ep} loss {loss.item():.3f} {(time.time()-t0)/60:.1f}min',flush=True)
net.eval(); out=CHECKPOINTS/'ssl_vits_thickdino_ft_chitt'; out.mkdir(parents=True,exist_ok=True)
torch.save(net.enc.state_dict(),out/'encoder.pth'); torch.save(net.head.state_dict(),out/'head.pth')
print('CHITT_FT_DONE',flush=True)
