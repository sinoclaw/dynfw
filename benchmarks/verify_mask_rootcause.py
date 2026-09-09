"""
决定性验证: v6 因果泄漏根因 = masked_fill(~causal, 0) 而非 -inf
复现 FWAttention 块内注意, 对比 mask_val=0 vs -inf 的过去泄漏量, 并检测 NaN。
"""
import torch

torch.manual_seed(2)
B, nh, D, N, W, T = 1, 4, 32, 128, 8, 48
Q = torch.randn(B, nh, T, N)
V = torch.randn(B, 1, T, D)

def chunk_attn(Q, K, V, W, mask_val):
    out = []
    for st in range(0, T, W):
        en = min(st + W, T)
        qc = Q[:, :, st:en]; kc = K[:, :, st:en]; vc = V[:, :, st:en]
        w = en - st
        sim = qc @ kc.mT
        causal = torch.tril(torch.ones(w, w, dtype=torch.bool), diagonal=-1)
        sim = sim.masked_fill(~causal, mask_val)
        attn = torch.softmax(sim, dim=-1)
        agg = attn @ vc
        out.append(agg)
    return torch.cat(out, dim=2)

for name, mv in [("mask=0 (原实现/泄漏)", 0.0), ("mask=-inf (修复/应无泄漏)", float('-inf'))]:
    out_A = chunk_attn(Q, Q, V, W, mv)
    V_B = V.clone(); V_B[0, 0, 12] += 1.0
    out_B = chunk_attn(Q, Q, V_B, W, mv)
    past = (out_A[:, :, :12] - out_B[:, :, :12]).abs().max().item()
    nan = torch.isnan(out_A).sum().item()
    print(f"{name}: 过去i<12 maxdiff={past:.2e} {'⚠泄漏' if past>1e-9 else 'OK'} | NaN数={nan}")
