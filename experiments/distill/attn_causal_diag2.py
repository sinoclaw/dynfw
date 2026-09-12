"""逐环节定位 v6 非因果泄漏 —— 用完整 BDHBlockFWCycle, 逐环节测 full vs cut diff.
环节: ln -> encoder -> relu -> attn -> yKV@encoder_v -> *y_sparse -> @decoder -> 残差
"""
import torch
import torch.nn.functional as F
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycle

torch.manual_seed(0)
D, nh, mlp_mult, vocab = 128, 4, 128, 256
W = 256; T = 12
blk = BDHBlockFWCycle(D, nh, mlp_mult, vocab, steps=1, W=W).eval()
N = blk.attn.N

x = torch.randn(1, 1, T, D)  # [B,1,T,D]

def fwd_all(x, store):
    """跑完整 block, 存各环节中间量到 store"""
    C = blk.config
    B, _, TT, _ = x.shape; nh = C.n_head
    x0 = blk.ln(x)
    x_latent = x0 @ blk.encoder                # [B,nh,T,N]
    x_sparse = F.relu(x_latent)
    yKV, new_mem = blk.attn(Q=x_sparse, K=x_sparse, V=x0, memories=None, W=W)
    yKV = blk.ln(yKV)
    y_latent = yKV @ blk.encoder_v
    y_sparse = F.relu(y_latent)
    xy = x_sparse * y_sparse
    yMLP = xy.transpose(1, 2).reshape(B, 1, TT, N * nh) @ blk.decoder
    y = blk.ln(yMLP)
    out = blk.ln(x0 + y)
    store['x0']=x0; store['x_sparse']=x_sparse; store['yKV']=yKV
    store['y_sparse']=y_sparse; store['y']=y; store['out']=out
    return out

s_full = {};  fwd_all(x, s_full)
print("环节 diff (full vs cut@t):")
for t in range(1, T):
    s_cut = {};  fwd_all(x[:, :, :t+1].contiguous(), s_cut)
    row = []
    for k in ['x0','x_sparse','yKV','y_sparse','y','out']:
        d = (s_full[k][0, :, t] - s_cut[k][0, :, t]).abs().max().item()
        row.append(f"{k}={d:.4f}")
    print(f"  t={t:2d}: " + "  ".join(row))
