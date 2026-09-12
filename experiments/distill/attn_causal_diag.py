"""最干净的 v6 因果性诊断 —— 直接测 FWAttention (带 causal mask 的那个), 逐位置。
目的是: 如果连 attn 层都非因果, 说明 causal mask 或 new_mem 有问题; 若 attn 因果,
则泄漏在 block 的其他环节 (encoder/decoder/ln/残差)。
"""
import torch
import torch.nn.functional as F
from dynfw.models.fused_fw_fw_cycle import FWAttention, Config

torch.manual_seed(0)
D, nh, N, mlp_mult, vocab = 128, 4, 512, 128, 256
cfg = Config(1, D, nh, mlp_mult, vocab)
attn = FWAttention(cfg).eval()
W = 256; T = 12

# 构造输入 (模拟 x_sparse, x)
x_sparse = torch.randn(1, nh, T, N)
x = torch.randn(1, 1, T, D)

def run(xs, xs_v):
    """xs: [B,nh,T,N] (K=Q), xs_v: [B,1,T,D] (V), 完整跑 attn"""
    return attn(Q=xs, K=xs, V=xs_v, memories=None, W=W)[0]  # [B,nh,T,D]

with torch.no_grad():
    out_full = run(x_sparse, x)
worst = 0
for t in range(1, T):
    with torch.no_grad():
        out_cut = run(x_sparse[:, :, :t+1], x[:, :, :t+1])
    d = (out_full[0, :, t] - out_cut[0, :, t]).abs().max().item()
    if d > worst: worst = d
    print(f"  t={t:2d} diff={d:.6f}")

print(f"\n== FWAttention 单独: 最大 diff={worst:.6f} -> {'因果✓' if worst<1e-5 else '非因果✗'} ==")

# 再看 causal mask 是否真的只用 tril (打印 attn forward 源码)
import inspect
src = inspect.getsource(attn.forward)
for line in src.split('\n'):
    if 'tril' in line or 'masked' in line or 'causal' in line or 'diagonal' in line:
        print("  [源码]", line.strip())
