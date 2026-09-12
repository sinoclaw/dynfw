"""diag_b32 的自我纠错版：消融必须全部在【同等带图】口径下比。

上一版把「eager 的 ①~⑤」和「带图的 修法」放一张表 → 脏对比。
本版：每个变体各自捕获一张图，只 replay 计时；并用 profiler 数 kernel 个数，
      交叉验证「两张图」没有少算东西（若 kernel 数与单张图相当 → 没少算）。
"""
import sys, time, argparse
import torch

sys.path.insert(0, '/data/dynfw')
from benchmarks.diag_b32 import build, DynAbl, DynDecoderS2, VOCAB, W


def capture_one(m, reps=7, warm=2):
    """给一个单图变体：捕获 + replay 计时 + 返回 (中位 ms, 图, 每步 kernel 数)"""
    with torch.no_grad():
        for _ in range(warm):
            m.step(); m.advance()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m.step()
        torch.cuda.synchronize()
        for _ in range(warm):
            g.replay(); m.advance()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.time(); g.replay(); m.advance(); torch.cuda.synchronize()
            ts.append(1000 * (time.time() - t0))
        k = kcount(lambda: (g.replay(), m.advance()))
    return sorted(ts)[len(ts) // 2], g, k


def kcount(fn, n=5):
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)


def prep(cls, B, T, seq, **kw):
    m = cls(build(), B=B)
    for k_, v in kw.items():
        setattr(m, k_, v)
    m.prefill(seq[:, :T]); m.tokbuf[:, 0] = seq[:, T]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--T', type=int, default=8192)
    a = ap.parse_args()
    B, T = a.batch, a.T
    torch.manual_seed(0)
    seq = torch.randint(0, VOCAB, (B, T + 64), device='cuda')

    print('=' * 96)
    print(f'【纠错版】全部变体同一口径 = 各自捕获成图后 replay    batch={B} T={T}')
    print('=' * 96)
    print(f'  {"变体":<40}{"ms/step":>10}{"kernel/step":>13}')

    abl = [
        ('单图 · 每步 fp32 fold + 写窗口 (=B′ 那张)', dict()),
        ('单图 · fold 改 bf16', dict(fold_dtype=torch.bfloat16)),
        ('单图 · 不算 fold（错的·仅诊断）', dict(skip_fold=True)),
        ('单图 · 不算 fold + 不写窗口（错的·下界）', dict(skip_fold=True, skip_where=True)),
    ]
    base = None
    for name, kw in abl:
        m = prep(DynAbl, B, T, seq, **kw)
        ms, g, k = capture_one(m)
        if base is None:
            base = ms
        print(f'  {name:<40}{ms:>10.2f}{k:>13d}')
        del m, g; torch.cuda.empty_cache()

    # 两张图分流
    m = prep(DynDecoderS2, B, T, seq)
    with torch.no_grad():
        m.do_fold = False
        for _ in range(3):
            m.step(); m.advance()
        g_no = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_no):
            m.step()
        m.do_fold = True
        m.prefill(seq[:, :T]); m.tokbuf[:, 0] = seq[:, T]
        for _ in range(3):
            m.step(); m.advance()
        g_fold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fold):
            m.step()
        m.prefill(seq[:, :T]); m.do_fold = True
        for _ in range(2):
            g_no.replay(); m.advance()
        torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            t0 = time.time()
            (g_fold if (m.pos % W == 0 and m.pos > 0) else g_no).replay()
            m.advance(); torch.cuda.synchronize()
            ts.append(1000 * (time.time() - t0))
        ms2 = sorted(ts)[len(ts) // 2]
        k_no = kcount(lambda: (g_no.replay(), m.advance()))
        k_fold = kcount(lambda: (g_fold.replay(), m.advance()))
    print(f'  {"两张图分流 · 非边界图":<40}{ms2:>10.2f}{k_no:>13d}')
    print(f'  {"两张图分流 · 边界图":<40}{"—":>10}{k_fold:>13d}')

    # 对手同场
    from benchmarks.bench_decode import TFDecoder, build_tf
    tfd = TFDecoder(build_tf(T + 64), B=B, maxlen=T + 64); tfd.prefill(seq[:, :T])
    with torch.no_grad():
        for _ in range(2):
            tfd.step(seq[:, tfd.pos])
        torch.cuda.synchronize(); ts = []
        for _ in range(7):
            t0 = time.time(); tfd.step(seq[:, tfd.pos]); torch.cuda.synchronize()
            ts.append(1000 * (time.time() - t0))
    tf = sorted(ts)[len(ts) // 2]
    print(f'  {"TF 原生动态(eager，同场)":<40}{tf:>10.2f}')
    print()
    print(f'  单图基线 {base:.2f}  →  两张图分流 {ms2:.2f}   =  分图收益 {base/ms2:.2f}x')
    print(f'  我们 {ms2:.2f} vs TF {tf:.2f}   =  {tf/ms2:.2f}x （>1 = 我们更快）')


if __name__ == '__main__':
    main()
