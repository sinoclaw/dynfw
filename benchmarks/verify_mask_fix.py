"""
修复方案验证 (简化): 方案A diagonal=0 + mask=-inf  vs 原实现 diag=-1+mask=0
判据: 过去无泄漏(<=0) 且 无NaN。
"""
import torch
torch.manual_seed(2)
B, nh, D, N, W, T = 1, 4, 32, 128, 8, 48
Q = torch.randn(B, nh, T, N)
V = torch.randn(B, 1, T, D)

def chunk_attn(Q, V, W, mask_val, diag):
    out = []
    for st in range(0, T, W):
        en = min(st + W, T)
        qc = Q[:, :, st:en]; vc = V[:, :, st:en]
        w = en - st
        sim = qc @ qc.mT   # K is Q
        causal = torch.tril(torch.ones(w, w, dtype=torch.bool), diagonal=diag)
        sim = sim.masked_fill(~causal, mask_val)
        attn = torch.softmax(sim, dim=-1)
        out.append(attn @ vc)
    return torch.cat(out, dim=2)

for name, mv, diag in [("方案A diag=0 + mask=-inf", float('-inf'), 0),
                       ("原实现 diag=-1 + mask=0", 0.0, -1)]:
    outA = chunk_attn(Q, V, W, mv, diag)
    V_B = V.clone(); V_B[0, 0, 12] += 1.0
    outB = chunk_attn(Q, V_B, W, mv, diag)
    past = (outA[:, :, :12] - outB[:, :, :12]).abs().max().item()
    nan = torch.isnan(outA).sum().item()
    leak = past > 1e-9 and past == past
    print(f"{name}: 过去i<12 maxdiff={past:.2e} {'⚠泄漏' if leak else 'OK-无泄漏'} | NaN={nan} {'⚠' if nan else 'OK'}")
