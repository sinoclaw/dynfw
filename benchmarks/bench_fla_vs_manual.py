"""C 方案的决定性对比：FLA 的 fused_chunk_simple_gla vs 我们的顺序循环（v6.6 现状）。

为什么必须同 bench 同形状：
  台账铁律「不跨口径比」「优化须真接进评测入口」。这里两边都只做
  "跨块状态更新 + 读出"这一段的等价工作，形状/T/预热/轮次完全一致。

我们的顺序循环 = v6.6 里 memories 的逐块更新（T/W 次串行），这是 C 要消灭的东西。
"""
import statistics
import time

import torch
import torch.nn.functional as F

DEV = 'cuda'


def manual_sequential(k, v, g, W=64, chunked_write=True):
    """v6.6 现状的最小复现：逐块串行更新 memory（T/W 次循环 + T/W 次小 kernel）。

    chunked_write=True 时模拟"块级门控 + 块内一次外积累加"（v6.6 的写法）；
    False 时模拟逐 token 更新（更接近朴素实现，用于上限估计）。
    """
    B, H, T, N = k.shape
    D = v.shape[-1]
    M = torch.zeros(B, H, N, D, device=k.device, dtype=k.dtype)
    outs = []
    if chunked_write:
        for i in range(T // W):
            sl = slice(i * W, (i + 1) * W)
            kc, vc, gc = k[:, :, sl], v[:, :, sl], g[:, :, sl]
            alpha = gc.mean(-1).exp()                                  # 块级标量门控
            M = alpha[:, :, None, None] * M + torch.einsum('bhn,bhd->bhnd', kc.sum(2), vc.sum(2))
            outs.append(M)
    else:
        for t in range(T):
            kt, vt, gt = k[:, :, t], v[:, :, t], g[:, :, t]
            M = gt.exp()[:, :, None, None] * M + torch.einsum('bhn,bhd->bhnd', kt, vt)
            outs.append(M)
    return torch.stack(outs, 2)


def bench(fn, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


torch.manual_seed(0)
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla

print('=== C 方案正面对比：跨块状态更新段（同形状/同预热/同轮次）===')
print(f'{"T":>7s} {"W":>5s} {"块数":>5s} {"FLA 前向":>11s} {"手写前向":>11s} {"前向加速":>9s} '
      f'{"FLA fwd+bwd":>12s} {"手写 fwd+bwd":>13s} {"训练加速":>9s}')

for T, W in ((8192, 64), (8192, 256), (16384, 64), (32768, 256)):
    B, H, N, D = 1, 16, 512, 128
    k = torch.randn(B, H, T, N, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=DEV, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, H, T, device=DEV))
    nchunk = T // W

    q = k.clone()

    # ---- FLA（head-second [B,T,H,K]）----
    fla_args = (q.permute(0, 2, 1, 3).contiguous(), k.permute(0, 2, 1, 3).contiguous(),
                v.permute(0, 2, 1, 3).contiguous(), g.permute(0, 2, 1).contiguous())

    def run_fla_fwd():
        return fused_chunk_simple_gla(*fla_args, output_final_state=True)

    def run_fla_train():
        qq = fla_args[0].clone().requires_grad_(True)
        kk = fla_args[1].clone().requires_grad_(True)
        vv = fla_args[2].clone().requires_grad_(True)
        gg = fla_args[3].clone().requires_grad_(True)
        o, _ = fused_chunk_simple_gla(qq, kk, vv, gg, output_final_state=True)
        o.float().sum().backward()

    def run_manual_fwd():
        return manual_sequential(k, v, g, W)

    def run_manual_train():
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        gg = g.clone().requires_grad_(True)
        out = manual_sequential(kk, vv, gg, W)
        out.float().sum().backward()

    try:
        f_fla = bench(run_fla_fwd)
        f_man = bench(run_manual_fwd, warmup=1, reps=3)
        t_fla = bench(run_fla_train, warmup=2, reps=3)
        t_man = bench(run_manual_train, warmup=1, reps=3)
        print(f'{T:>7d} {W:>5d} {nchunk:>5d} {f_fla:>10.1f}ms {f_man:>10.1f}ms '
              f'{f_man/f_fla:>8.1f}x {t_fla:>11.1f}ms {t_man:>12.1f}ms {t_man/t_fla:>8.1f}x')
    except Exception as e:
        print(f'{T:>7d} {W:>5d} {nchunk:>5d}  FAIL {type(e).__name__}: {str(e)[:80]}')

print()
print('注：手写版只做"跨块状态更新"（不含块内注意），是 C 要替换的那一段的最小复现。')
print('    加速比 = 手写/FLA，>1 表示 FLA 快。')
