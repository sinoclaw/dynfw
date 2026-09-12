"""OPT5（手写块内注意 + 前缀和）J1 正确性 + J2 性能逐条取数。

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_opt5.py --Ts 1024,4096,8192,16384 --batch 2
"""
import sys, time, argparse
import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt2, to_opt5
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


def causality(model, T, POS, seed=0):
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T + 2,), device='cuda')
    with torch.no_grad():
        model.eval()
        l1 = model(seg[:POS + 1].unsqueeze(0), None)[0][0, POS, :].float()
        l2 = model(seg[:POS + 2].unsqueeze(0), None)[0][0, POS, :].float()
    return (l1 - l2).abs().max().item()


def numerics(m_base, m_opt, Ts=(256, 1024)):
    out = []
    for T in Ts:
        torch.manual_seed(7)
        x = torch.randint(0, VOCAB, (1, T), device='cuda')
        with torch.no_grad():
            m_base.eval(); m_opt.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = m_base(x, None)[0].float()
                b = m_opt(x, None)[0].float()
        md = (a - b).abs().max().item()
        sc = a.abs().max().item()
        out.append((T, md, sc, md / max(sc, 1e-9)))
    return out


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
        torch.cuda.empty_cache(); return None, None, f'{type(e).__name__}: {str(e)[:60]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='1024,4096,8192,16384,32768')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--skip-j1', action='store_true')
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    if not a.skip_j1:
        print('=' * 88)
        print('J1 正确性（OPT5 / OPT5-bf16prefix）')
        print('=' * 88)
        m_base = build_base().cuda()
        variants = [
            ('OPT2 前缀和(现行)', to_opt2(build_base(), True)),
            ('OPT5 手写注意', to_opt5(build_base(), True, False)),
            ('OPT5 bf16前缀和', to_opt5(build_base(), True, True)),
        ]
        for name, m in variants:
            m = m.cuda()
            cd = max(causality(m, T=40, POS=30), causality(m, T=80, POS=60),
                     causality(m, T=300, POS=290))
            tag = 'CAUSAL OK' if cd < 1e-6 else 'LEAK!'
            print(f'  因果探针 {name:<20} maxdiff = {cd:.6e}  {tag}')
        print()
        for name, m in variants:
            for T, md, sc, rel in numerics(m_base, m):
                print(f'  数值 {name:<20} T={T:<5} maxdiff={md:.4e}  相对={rel:.3e}  (幅值{sc:.2f})')
        print()

    print('=' * 88)
    print(f'J2 性能（batch={a.batch}，ms/step 越低越好）')
    print('=' * 88)
    cols = [('TF', build_tf), ('基线', build_base),
            ('OPT2', lambda: to_opt2(build_base(), True)),
            ('OPT5', lambda: to_opt5(build_base(), True, False)),
            ('OPT5b16', lambda: to_opt5(build_base(), True, True)),
            ('OPT5+comp', lambda: torch.compile(to_opt5(build_base(), True, False)))]
    print(f"{'T':>7}" + ''.join(f'{n:>14}' for n, _ in cols))
    for T in Ts:
        row = f'{T:>7}'
        for name, mk in cols:
            try:
                m = mk().cuda()
            except Exception as e:
                row += f"{'buildERR':>14}"; continue
            dt, mem, err = bench(m, T, a.batch)
            row += f"{(f'{dt*1e3:.1f}ms' if dt else '--'):>14}"
            del m; torch.cuda.empty_cache()
        print(row)


if __name__ == '__main__':
    main()
