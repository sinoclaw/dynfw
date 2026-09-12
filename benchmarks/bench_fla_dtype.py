"""验证假设：FLA 的 fused_chunk_simple_gla 在 fp32 下比 bf16 慢一个量级。

若成立 ⇒ 提速方案 = FLA 段走 bf16（前向反向都快），模型其余部分保持 fp32。
口径：同形状（贴 v6.7 真实配置: T=8192, H=16, K=N=512, V=D=128）、同预热、多轮取中位。
"""
import statistics
import time

import torch
import torch.nn.functional as F

DEV = 'cuda'
torch.manual_seed(0)

from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla

B, T, H, K, V = 1, 8192, 16, 512, 128


def bench(dtype, chunk_size=None, warmup=3, reps=7):
    q = torch.randn(B, T, H, K, device=DEV, dtype=dtype)
    k = torch.randn(B, T, H, K, device=DEV, dtype=dtype)
    v = torch.randn(B, T, H, V, device=DEV, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV, dtype=torch.float32))
    if dtype == torch.bfloat16:
        g = g.to(torch.bfloat16)
    kw = {}
    if chunk_size is not None:
        kw['chunk_size'] = chunk_size

    def fwd():
        return fused_chunk_simple_gla(q, k, v, g, output_final_state=True, **kw)

    def train():
        qq = q.clone().requires_grad_(True); kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True); gg = g.clone().requires_grad_(True)
        o, _ = fused_chunk_simple_gla(qq, kk, vv, gg, output_final_state=True, **kw)
        o.float().sum().backward()

    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()
    fts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fwd()
        torch.cuda.synchronize(); fts.append((time.perf_counter() - t0) * 1000)

    for _ in range(2):
        train()
    torch.cuda.synchronize()
    bts = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter(); train()
        torch.cuda.synchronize(); bts.append((time.perf_counter() - t0) * 1000)

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    del q, k, v, g
    torch.cuda.empty_cache()
    return statistics.median(fts), statistics.median(bts), peak


print(f'=== FLA fused_chunk_simple_gla: fp32 vs bf16（T={T}, H={H}, K={K}, V={V}）===')
print(f'{"dtype":10s} {"chunk":>7s} {"前向":>10s} {"fwd+bwd":>11s} {"反向占比":>9s} {"peak":>9s}')
res = {}
for name, dt in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
    for cs in (None, 128, 256):
        try:
            f, b, pk = bench(dt, cs)
            tag = f'{name}'
            res[(name, cs)] = (f, b)
            print(f'{tag:10s} {str(cs):>7s} {f:9.2f}ms {b:10.2f}ms {(b-f)/b*100:8.1f}% {pk:8.2f}GiB', flush=True)
        except Exception as e:
            print(f'{name:10s} {str(cs):>7s} FAIL: {type(e).__name__}: {str(e)[:70]}', flush=True)

print()
if ('fp32', None) in res and ('bf16', None) in res:
    f32, b32 = res[('fp32', None)], res[('bf16', None)]
    print(f'  ⟹ fp32/bf16 训练耗时比 = {b32[1]/f32[1]:.2f}x' if b32[1] > f32[1] else f'  ⟹ fp32 更快')
    print(f'     前向 {f32[0]:.2f}ms vs {b32[0]:.2f}ms（{f32[0]/b32[0]:.2f}x）')
    print(f'     训练 {f32[1]:.2f}ms vs {b32[1]:.2f}ms（{f32[1]/b32[1]:.2f}x）')
print()
print('注：若 bf16 明显更快，则提速方案 = FLA 段用 bf16（其余保持 fp32），需另验能力不损。')
