"""公平桌 v3：修掉 recompile 限制 + 编译版因果探针扫 T（判噪声 vs 真泄漏）。

判据（跑前锁死）：
  J1  因果：OPT5+comp 若随 T 增大而放大 → 真泄漏；若持平在 fp32 噪声量级(≤1e-5) → 归约顺序差，良性。
  J2  公平：compile 必须同等发给 TF 与基线；判据 = OPT5+comp vs TF+comp。
用法: PYTHONPATH=/data/dynfw python benchmarks/bench_fair3.py --Ts 1024,4096,8192,16384,32768 --batch 2
"""
import sys, time, argparse
import torch

# 关键修复：J1 的多 shape 探针会把 dynamo 的 recompile 预算吃满 → J2 静默回退 eager
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5, to_opt2
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


def build_opt5():
    return to_opt5(build_base(), True, False)


def causality(model, T, nlast=10, seed=0):
    """长度依赖探针：T 与 T+1 序列在倒数第 nlast 位的 logits 必须一致。"""
    POS = T - nlast
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T + 2,), device='cuda')
    with torch.no_grad():
        model.eval()
        l1 = model(seg[:POS + 1].unsqueeze(0), None)[0][0, POS, :].float()
        l2 = model(seg[:POS + 2].unsqueeze(0), None)[0][0, POS, :].float()
    return (l1 - l2).abs().max().item(), l1.abs().max().item()


def bench(m, T, B=2, iters=3, warmup=3):
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


COLS = [
    ('TF', build_tf, False),
    ('TF+comp', build_tf, True),
    ('基线', build_base, False),
    ('基线+comp', build_base, True),
    ('OPT2+comp', lambda: to_opt2(build_base(), True), True),
    ('OPT5+comp', build_opt5, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='1024,4096,8192,16384,32768')
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    print('=' * 96)
    print('J1 因果探针扫 T —— 真泄漏随 T 放大；归约顺序差持平在 fp32 噪声')
    print('=' * 96)
    for name, mk, comp in [('OPT5', build_opt5, False), ('OPT5+comp', build_opt5, True),
                           ('TF', build_tf, False), ('TF+comp', build_tf, True)]:
        m = mk()
        if comp:
            m = torch.compile(m)
        m = m.cuda()
        print(f'  --- {name} ---')
        for T in (40, 80, 160, 320, 640, 1280, 2560):
            md, sc = causality(m, T)
            flag = 'ok' if md < 1e-5 else 'CHECK'
            print(f'      T={T:<6} maxdiff={md:.6e}  (logits幅值 {sc:.3f})  {flag}')
        del m; torch.cuda.empty_cache()
    print()

    print('=' * 96)
    print(f'J2 公平对照 v3（batch={a.batch}，cache_size_limit=1000，compile 同等对待）')
    print('=' * 96)
    print(f"{'T':>7}" + ''.join(f'{n:>13}' for n, _, _ in COLS))
    res = {}
    for T in Ts:
        row = f'{T:>7}'
        for name, mk, comp in COLS:
            m = mk()
            if comp:
                m = torch.compile(m)
            m = m.cuda()
            dt, mem, err = bench(m, T, a.batch)
            res[(name, T)] = dt
            row += f"{(f'{dt*1e3:.1f}ms' if dt else '--'):>13}"
            del m; torch.cuda.empty_cache()
        print(row)
    print()
    print('关键判据 —— 相对 TF（未编译） / 相对 TF+comp（公平桌）:')
    for T in Ts:
        tf, tfc = res.get(('TF', T)), res.get(('TF+comp', T))
        o5, o5c = res.get(('OPT5+comp', T)), res.get(('OPT5+comp', T))
        s = f'  T={T:<6}'
        if tf and o5c:
            s += f' OPT5+comp/TF = {tf/o5c:5.2f}x'
        if tfc and o5c:
            s += f'   OPT5+comp/TF+comp = {tfc/o5c:5.2f}x'
        print(s)


if __name__ == '__main__':
    main()
