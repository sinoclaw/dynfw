"""对比 LM 封装层 vs block 层 —— 定位非因果是 block 内还是 LM 头/embed 引入。
关键: block 层单独 = 严格因果(已证), 看 LM 完整(含 head) 是否冒出非因果。
"""
import torch
import torch.nn.functional as F
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM, BDHBlockFWCycle

torch.manual_seed(0)
D, nh, mlp_mult, vocab = 128, 16, 64, 256
W = 256; T = 12; n_layer = 2

# LM 完整
lm = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W).eval()
ids = torch.randint(0, vocab, (1, T))

with torch.no_grad():
    lg_full = lm.forward(ids, None)[0]  # [1,T,V]
worst = 0
for t in range(1, T):
    with torch.no_grad():
        lg_cut = lm.forward(ids[:, :t+1], None)[0]
    d = (lg_full[0, t] - lg_cut[0, t]).abs().max().item()
    if d > worst: worst = d
print(f"[LM完整 n_layer={n_layer}] 最大 diff={worst:.6f} -> {'因果✓' if worst<1e-5 else '非因果✗'}")

# n_layer=1 对比
lm1 = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=1, steps=1, mlp_mult=mlp_mult, W=W).eval()
with torch.no_grad():
    lg_full1 = lm1.forward(ids, None)[0]
worst1 = 0
for t in range(1, T):
    with torch.no_grad():
        lg_cut1 = lm1.forward(ids[:, :t+1], None)[0]
    d = (lg_full1[0, t] - lg_cut1[0, t]).abs().max().item()
    if d > worst1: worst1 = d
print(f"[LM完整 n_layer=1] 最大 diff={worst1:.6f} -> {'因果✓' if worst1<1e-5 else '非因果✗'}")

# 只 block 层但用 LM 的 blocks[0] 和 embed
with torch.no_grad():
    h = lm.ln(lm.e(ids)).unsqueeze(1)
    b0 = lm.blocks[0]
    out_full, _ = b0(h, None)
worstb = 0
for t in range(1, T):
    with torch.no_grad():
        out_cut, _ = b0(h[:, :, :t+1].contiguous(), None)
    d = (out_full[0, :, t] - out_cut[0, :, t]).abs().max().item()
    if d > worstb: worstb = d
print(f"[LM.blocks[0] (用lm的embed输入)] 最大 diff={worstb:.6f} -> {'因果✓' if worstb<1e-5 else '非因果✗'}")
