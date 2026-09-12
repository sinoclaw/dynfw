"""(a) 稳态速度重测：排除 Triton 首次 JIT 编译；(b) 诊断 o 的差异来源。

case 4 首测 45s 几乎肯定是编译开销，必须 warmup 后多轮取中位才有意义（这也呼应台账里
"torch.compile 读数取决于编译时机 —— 编译前必须 warmup"的教训）。
"""
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
print('[a] 稳态速度（warmup 3 轮 + 5 轮取中位）')
print(f'    {"配置":34s} {"前向":>10s} {"反向":>10s} {"peak":>9s}')

configs = [
    ('T=8192  H=16 K=512 V=128 (我们真实形状)', 1, 8192, 16, 512, 128),
    ('T=8192  H=16 K=128 V=128 (K 缩小 4x)',    1, 8192, 16, 128, 128),
    ('T=8192  H=16 K=512 V=128 (重复，验稳定性)', 1, 8192, 16, 512, 128),
    ('T=16384 H=16 K=512 V=128',               1, 16384, 16, 512, 128),
]

import statistics
for name, B, T, H, K, V in configs:
    q = torch.randn(B, T, H, K, device=DEV)
    k = torch.randn(B, T, H, K, device=DEV)
    v = torch.randn(B, T, H, V, device=DEV)
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV))
    for _ in range(3):                                   # warmup（含 JIT 编译）
        o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)
    torch.cuda.synchronize()
    fwds = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)
        torch.cuda.synchronize(); fwds.append((time.perf_counter() - t0) * 1000)
    torch.cuda.reset_peak_memory_stats()
    qq = q.clone().requires_grad_(True); kk = k.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True); gg = g.clone().requires_grad_(True)
    for _ in range(2):
        oo, _ = fused_chunk_simple_gla(qq, kk, vv, gg, output_final_state=True)
        oo.float().sum().backward()
    torch.cuda.synchronize()
    bwds = []
    for _ in range(3):
        oo, _ = fused_chunk_simple_gla(qq, kk, vv, gg, output_final_state=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        oo.float().sum().backward()
        torch.cuda.synchronize(); bwds.append((time.perf_counter() - t0) * 1000)
    print(f'    {name:34s} {statistics.median(fwds):9.1f}ms {statistics.median(bwds):9.1f}ms '
          f'{torch.cuda.max_memory_allocated()/2**30:8.2f}GiB')

# ---------- (b) o 的差异诊断 ----------
print()
print('[b] o 差异诊断（首测 maxdiff 73 / rel 0.88；state 却只有 9.4e-3）')
B, T, H, K, V = 2, 128, 4, 32, 16
q = torch.randn(B, T, H, K, device=DEV)
k = torch.randn(B, T, H, K, device=DEV)
v = torch.randn(B, T, H, V, device=DEV)
g = F.logsigmoid(torch.randn(B, T, H, device=DEV))
o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)


def naive(after=True):
    S = torch.zeros(B, H, K, V, device=DEV, dtype=torch.float32)
    outs = []
    for t in range(T):
        if not after:
            outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t].float(), S))
        S = torch.exp(g[:, t]).float().unsqueeze(-1).unsqueeze(-1) * S \
            + k[:, t].float().unsqueeze(-1) * v[:, t].float().unsqueeze(-2)
        if after:
            outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t].float(), S))
    return torch.stack(outs, 1)


print(f'    o  量级: FLA |o|max={o.abs().max():.3f}   朴素 |o|max={naive().abs().max():.3f}')
on = naive()
ratio = (o.float().abs().mean(2, keepdim=True) / on.abs().mean(2, keepdim=True) + 1e-9)
print(f'    逐位置比例 (FLA/朴素): 中位={ratio.median():.4f}  范围 {ratio.min():.4f}~{ratio.max():.4f}')
print(f'    ⇒ 若比例近似常数 ⇒ 只差一个 scale；否则是读出方式本质不同')
# 尝试：o 是否等于 "q_t @ S_{t-1}" 的某种缩放
on_pre = naive(after=False)
for tag, cand in (('写后', on), ('写前', on_pre)):
    d = (o.float() - cand).abs().max().item()
    sc = (o.float() * cand).sum() / (cand * cand).sum()      # 最小二乘 scale
    print(f'    假设[{tag}]: maxdiff={d:.3e}  最小二乘最优 scale={sc:.4f}')
print()
print('    结论：state 已完全对齐 ⇒ 记忆更新公式一致；o 的读出方式我们本来就不用')
print('    （我们块内走自家 raw 读出，块间只取 FLA 的 state）')
