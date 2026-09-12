"""方案 A 验证：融合实现 vs 基线 vs TF —— 按 PLAN-A-FUSED-IMPL.md 的判据逐条取数。

J1 正确性: 因果探针 maxdiff 必须 =0；同权重 logits maxdiff
J2 性能  : T=1024/4096/8192/16384 的 ms/step（对比 TF）
用法: PYTHONPATH=/data/dynfw python benchmarks/bench_opt_vs_tf.py
"""
import sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt, to_opt2, to_opt3
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


# ---------- J1 ----------
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


# ---------- J2 ----------
def bench(m, T, B=2, iters=3, warmup=1):
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
        torch.cuda.empty_cache(); return None, None, str(e)[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='1024,4096,8192,16384')
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    print('=' * 82)
    print('J1 正确性')
    print('=' * 82)
    m_base = build_base().cuda()
    m_cons = to_opt(build_base(), strict_bf16=False).cuda()   # 保守版（保留 fp32 转换）
    m_bf16 = to_opt(build_base(), strict_bf16=True).cuda()    # 全 bf16 版
    m_p2 = to_opt2(build_base(), strict_bf16=True).cuda()     # 并行前缀和版
    m_p3 = to_opt3(build_base(), strict_bf16=True).cuda()     # +LN拉回bf16 版

    for name, m in [('基线', m_base), ('OPT1 保守(保留fp32)', m_cons),
                    ('OPT1 全bf16', m_bf16), ('OPT2 并行前缀和', m_p2),
                    ('OPT3 +LN回bf16', m_p3)]:
        try:
            cd = max(causality(m, T=40, POS=30), causality(m, T=80, POS=60))
            print(f"  因果探针 {name:<22} maxdiff = {cd:.6e}  {'CAUSAL OK' if cd < 1e-6 else 'LEAK!'}")
        except Exception as e:
            print(f"  因果探针 {name:<22} ERROR {type(e).__name__}: {str(e)[:60]}")
    print()
    for name, m in [('OPT1 保守(保留fp32)', m_cons), ('OPT1 全bf16', m_bf16),
                    ('OPT2 并行前缀和', m_p2),
                    ('OPT3 +LN回bf16', m_p3)]:
        for T, md, sc, rel in numerics(m_base, m):
            print(f"  数值 {name:<22} T={T:<5} maxdiff={md:.4e}  相对={rel:.3e}  (logits幅值{sc:.2f})")
    print()

    print('=' * 82)
    print(f'J2 性能（batch={a.batch}，ms/step 越低越好）')
    print('=' * 82)
    hdr = f"{'T':>7}" + ''.join(f"{n:>17}" for n in ('TF', '基线 v6w', 'OPT2 前缀和', 'OPT3 +LN回bf16'))
    print(hdr)
    for T in Ts:
        row = f"{T:>7}"
        for name, mk in (('tf', build_tf), ('base', lambda: build_base()),
                         ('p2', lambda: to_opt2(build_base(), True)),
                         ('p3', lambda: to_opt3(build_base(), True))):
            m = mk().cuda()
            dt, mem, err = bench(m, T, a.batch)
            row += f"{(f'{dt*1e3:.1f}ms' if dt else '--'):>17}"
            del m; torch.cuda.empty_cache()
        print(row)


if __name__ == '__main__':
    main()
