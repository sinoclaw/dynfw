"""A″ 收尾：① CUDA Graphs（reduce-overhead）② OPT5+comp 残余开销的真实 kernel 名。

用法: PYTHONPATH=/data/dynfw python benchmarks/opt5_final.py
"""
import sys, time
import torch
from torch.profiler import profile, ProfilerActivity

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def mk(kind, mode=None):
    if kind == 'tf':
        m = TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)
    else:
        torch.manual_seed(0)
        m = to_opt5(BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL,
                                      steps=1, mlp_mult=4, W=W), True, False)
    if mode:
        m = torch.compile(m, mode=mode)
    return m


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)


def bench_median(m, T, B=2, rounds=3, iters=3, warmup=4):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(warmup):
        run_step(m, opt, x, y)
    torch.cuda.synchronize()
    s = []
    for _ in range(rounds):
        t0 = time.time()
        for _ in range(iters):
            run_step(m, opt, x, y)
        torch.cuda.synchronize()
        s.append(1000 * (time.time() - t0) / iters)
    s.sort()
    return s[len(s) // 2]


def main():
    print('=' * 92)
    print('A 残余 kernel 归因（OPT5+comp, T=8192, batch=2）')
    print('=' * 92)
    m = mk('opt5', 'default').cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    B, T = 2, 8192
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(3):
        run_step(m, opt, x, y)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        run_step(m, opt, x, y); torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.key, e.count) for e in p.key_averages()
            if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f'  {"pct":>6}{"ms":>9}{"calls":>7}  kernel')
    for us, k, c in rows[:20]:
        print(f'  {us/tot:>6.1%}{us/1000:>9.2f}{c:>7}  {k[:70]}')
    del m, opt, x, y; torch.cuda.empty_cache()

    print()
    print('=' * 92)
    print('B CUDA Graphs（reduce-overhead）/ max-autotune 对照，中位×3')
    print('=' * 92)
    combos = [(k, mo, lab) for k in ('tf', 'opt5')
              for mo, lab in ((None, 'eager'), ('default', 'default'),
                              ('reduce-overhead', 'reduce-oh'),
                              ('max-autotune-no-cudagraphs', 'max-auto'))]
    for T in (8192, 32768):
        print(f'  --- T={T} batch={B} ---')
        base = None
        for kind, mode, label in combos:
            try:
                mm = mk(kind, mode).cuda()
                med = bench_median(mm, T, B)
                if kind == 'tf' and label == 'eager':
                    base = med
                ratio = f'  (vs TF eager {base/med:.2f}x)' if base else ''
                print(f'      {kind:<5} {label:<10} {med:8.1f} ms{ratio}')
                del mm; torch.cuda.empty_cache()
            except Exception as e:
                print(f'      {kind:<5} {label:<10} ERR {type(e).__name__}: {str(e)[:50]}')
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
