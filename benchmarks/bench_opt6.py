"""OPT6/OPT7（让 Flash 可用）—— 后端探针 + J1 等价性 + 公平桌。

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_opt6.py
"""
import sys, time, argparse
import torch
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5, to_opt6, to_opt7
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


MK = {
    'OPT5': lambda: to_opt5(build_base(), True, False),
    'OPT6': lambda: to_opt6(build_base(), True, False),
    'OPT7': lambda: to_opt7(build_base(), True, False),
    'TF': build_tf,
}


def prob_backends():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print('=' * 92)
    print('Part A SDPA 后端可用性探针（显式指定后端，失败即不可用）')
    print('=' * 92)
    cases = [('q/k=128, v=128  (OPT6 切份后)', 128, 128),
             ('q/k=128, v=256  (原始，已知不可用)', 128, 256),
             ('q/k=256, v=256  (OPT7 填充后)', 256, 256)]
    for name, nd, vd in cases:
        # ⚠️ 必须 4-D：【坑】3-D 输入会让 FLASH/EFFIC 全部报告不可用（我第一版就栽在这）
        q = torch.randn(8, 1, 256, nd, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(8, 1, 256, vd, device='cuda', dtype=torch.bfloat16)
        line = f'  {name:<38}'
        for bn, be in [('FLASH', SDPBackend.FLASH_ATTENTION),
                       ('EFFIC', SDPBackend.EFFICIENT_ATTENTION),
                       ('MATH', SDPBackend.MATH)]:
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, q, v, is_causal=True, scale=1.0)
                line += f' {bn}:OK'
            except Exception as e:
                line += f' {bn}:NO'
        print(line)
        del q, v; torch.cuda.empty_cache()


def caus_fixedlen(model, T=1024, positions=(13, 519, 1020), seed=0):
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T,), device='cuda')
    worst, sc = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for POS in positions:
            if POS + 1 >= T:
                continue
            x1 = seg.clone(); x1[POS + 1:] = 7
            x2 = seg.clone(); x2[POS + 1:] = 123
            l1 = model(x1.unsqueeze(0), None)[0][0, POS, :].float()
            l2 = model(x2.unsqueeze(0), None)[0][0, POS, :].float()
            worst = max(worst, (l1 - l2).abs().max().item())
            sc = max(sc, l1.abs().max().item())
    return worst, sc


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


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return loss


def bench_median(m, T, B=2, rounds=3, iters=3, warmup=3):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='8192,16384,32768')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--part', default='abc')
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    if 'a' in a.part:
        prob_backends()

    if 'b' in a.part:
        print()
        print('=' * 92)
        print('Part B J1 等价性（同形状因果探针 + 与基线 logits 对照）')
        print('=' * 92)
        m_base = build_base().cuda()
        for name in ('OPT5', 'OPT6', 'OPT7'):
            m = MK[name]().cuda()
            try:
                cd, sc = caus_fixedlen(m, 1024)
                print(f'  {name:<6} 因果 maxdiff={cd:.6e} (幅值{sc:.3f}) '
                      f'{"PASS" if cd < 1e-5 else "CHECK"}')
                for T, md, s, rel in numerics(m_base, m):
                    print(f'        数值 T={T:<5} maxdiff={md:.4e} 相对={rel:.3e} (幅值{s:.2f})')
            except Exception as e:
                print(f'  {name:<6} ERROR {type(e).__name__}: {str(e)[:70]}')
            del m; torch.cuda.empty_cache()

    if 'c' in a.part:
        print()
        print('=' * 92)
        print(f'Part C 公平桌（batch={a.batch}，compile 同等对待，中位×3）')
        print('=' * 92)
        cols = ['TF', 'OPT5', 'OPT6', 'OPT7']
        print(f"{'T':>7}" + ''.join(f'{c+chr(43)+"comp":>15}' for c in cols))
        res = {}
        for T in Ts:
            row = f'{T:>7}'
            for c in cols:
                try:
                    m = torch.compile(MK[c]()).cuda()
                    med = bench_median(m, T, a.batch)
                    res[(c, T)] = med
                    row += f'{f"{med:.1f}ms":>15}'
                    del m; torch.cuda.empty_cache()
                except Exception as e:
                    row += f'{"ERR":>15}'
                    torch.cuda.empty_cache()
                    print(f'   !! {c} T={T}: {type(e).__name__} {str(e)[:60]}')
            print(row)
        print()
        for T in Ts:
            tf = res.get(('TF', T))
            if not tf:
                continue
            line = f'  T={T:<6}'
            for c in ('OPT5', 'OPT6', 'OPT7'):
                v = res.get((c, T))
                if v:
                    line += f'  {c}/TF={tf/v:5.2f}x'
            print(line)


if __name__ == '__main__':
    main()
