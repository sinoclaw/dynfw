"""扫 T 测墙钟斜率 —— 判「我们的 O(T)」是否真存在（军规判据：O(T)≈1.2 / O(T²)≈1.7）。

为什么必须做这个：TinyStories 那个实验 T=1024 / W=256 只有 4 个块，
块内仍是 256×256 全注意 —— 我们和 TF 的实际计算量几乎一样(≈1.0e11 FLOPs/forward)，
测的是「谁的内核更快」而不是「谁的复杂度更低」。本脚本给出真正的判据。

同时输出：每步耗时、峰值显存、log-log 斜率。
用法: PYTHONPATH=/data/dynfw python benchmarks/sweep_T_slope.py
"""
import sys, time, argparse, math
import torch, torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')

D = 256
NH = 8
NL = 6
VOCAB = 50257


def build(arch, W=256):
    if arch == 'v6w':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL)
    raise ValueError(arch)


def bench(m, T, B, device, iters=3, warmup=1):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    def one():
        x = torch.randint(0, VOCAB, (B, T), device=device)
        y = torch.randint(0, VOCAB, (B, T), device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    try:
        for _ in range(warmup):
            one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        return dt, torch.cuda.max_memory_allocated() / 1e9, None
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return None, None, 'OOM'
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return None, None, str(e)[:60]


def slope(ts, ys):
    """log-log 最小二乘斜率"""
    xs = [math.log(t) for t in ts]
    ls = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ls) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ls))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=4)
    a = ap.parse_args()
    dev = a.device
    Ts = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    for arch, W in [('tf', None), ('v6w', 256), ('v6w', 1024)]:
        label = arch if W is None else f'{arch}(W={W})'
        print(f"\n{'='*74}\n### {label}   batch={a.batch}  D={D} nh={NH} L={NL}\n{'='*74}")
        print(f"{'T':>8}{'秒/步':>12}{'峰值显存GB':>14}{'相对T=512':>12}{'备注':>10}")
        ts, ys = [], []
        base = None
        for T in Ts:
            m = build(arch, W) if W else build(arch)
            m = m.to(dev)
            dt, mem, err = bench(m, T, a.batch, dev)
            if err:
                print(f"{T:>8}{'--':>12}{'--':>14}{'--':>12}{err:>10}")
                del m; torch.cuda.empty_cache()
                continue
            if base is None:
                base = dt
            ts.append(T); ys.append(dt)
            print(f"{T:>8}{dt*1e3:>10.1f}ms{mem:>12.2f}   {dt/base:>10.2f}x{'':>10}")
            del m; torch.cuda.empty_cache()
        if len(ts) >= 3:
            s = slope(ts, ys)
            verdict = 'O(T) 线性 ✓' if s < 1.45 else ('O(T^2) 二次 ✗' if s > 1.55 else '不确定')
            print(f"--- log-log 斜率 = {s:.3f}  ->  {verdict}  (军规: O(T)≈1.2 / O(T^2)≈1.7) ---")


if __name__ == '__main__':
    main()
