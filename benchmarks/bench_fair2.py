"""公平对照：把 torch.compile 同等发给 TF 和基线，再比。

红线：编译优化只能作为「实现形态」变量，绝不能让一边独占 → 否则是脏评测。

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_fair2.py --Ts 1024,4096,8192,16384,32768 --batch 2
"""
import sys, time, argparse
import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


def bench(m, T, B=2, iters=3, warmup=2):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    try:
        for _ in range(warmup):
            one()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9, None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); return None, None, 'OOM'
    except Exception as e:
        torch.cuda.empty_cache(); return None, None, f'{type(e).__name__}: {str(e)[:50]}'


def causality(model, T, POS, seed=0):
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T + 2,), device='cuda')
    with torch.no_grad():
        model.eval()
        l1 = model(seg[:POS + 1].unsqueeze(0), None)[0][0, POS, :].float()
        l2 = model(seg[:POS + 2].unsqueeze(0), None)[0][0, POS, :].float()
    return (l1 - l2).abs().max().item()


COLS = [
    ('TF', lambda: build_tf()),
    ('TF+comp', lambda: torch.compile(build_tf())),
    ('基线', lambda: build_base()),
    ('基线+comp', lambda: torch.compile(build_base())),
    ('OPT5', lambda: to_opt5(build_base(), True, False)),
    ('OPT5+comp', lambda: torch.compile(to_opt5(build_base(), True, False))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='1024,4096,8192,16384,32768')
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    print('=' * 96)
    print('J1 正确性（编译版也要过因果探针）')
    print('=' * 96)
    for name, mk in [('OPT5', lambda: to_opt5(build_base(), True, False)),
                     ('OPT5+comp', lambda: torch.compile(to_opt5(build_base(), True, False)))]:
        m = mk().cuda()
        cd = max(causality(m, 40, 30), causality(m, 80, 60), causality(m, 300, 290))
        print(f'  因果探针 {name:<14} maxdiff = {cd:.6e}  {"CAUSAL OK" if cd < 1e-6 else "LEAK!"}')
        del m; torch.cuda.empty_cache()
    print()

    print('=' * 96)
    print(f'J2 公平对照（batch={a.batch}）—— compile 同等发给每一列')
    print('=' * 96)
    print(f"{'T':>7}" + ''.join(f'{n:>15}' for n, _ in COLS))
    res = {}
    for T in Ts:
        row = f'{T:>7}'
        for name, mk in COLS:
            try:
                m = mk().cuda()
            except Exception as e:
                row += f"{'buildERR':>15}"; continue
            dt, mem, err = bench(m, T, a.batch)
            res[(name, T)] = dt
            row += f"{(f'{dt*1e3:.1f}ms' if dt else '--'):>15}"
            del m; torch.cuda.empty_cache()
        print(row)
    print()
    print('相对 TF（>1 = 我们更快）:')
    for T in Ts:
        tf = res.get(('TF', T))
        if not tf:
            continue
        parts = []
        for name in ('TF+comp', '基线+comp', 'OPT5', 'OPT5+comp'):
            v = res.get((name, T))
            parts.append(f'{name}={tf/v:.2f}x' if v else f'{name}=--')
        print(f'  T={T:<6} ' + '  '.join(parts))
    print()
    print('相对 TF+comp（>1 = 我们更快；这是「公平桌」的判据）:')
    for T in Ts:
        tfc = res.get(('TF+comp', T))
        if not tfc:
            continue
        parts = []
        for name in ('OPT5', 'OPT5+comp', '基线+comp'):
            v = res.get((name, T))
            parts.append(f'{name}={tfc/v:.2f}x' if v else f'{name}=--')
        print(f'  T={T:<6} ' + '  '.join(parts))


if __name__ == '__main__':
    main()
