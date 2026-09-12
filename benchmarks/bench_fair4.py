"""公平桌 v4（精简编译量）。

因果探针改成「同形状」版：长度固定 T，只改 POS 之后的 token。
  真因果 ⇒ POS 处 logits 必须逐位相同（与未来内容无关）。
  → 单一 shape，编译一次即可；且比对「长度依赖法」更严格（不掺不同 shape 的归约顺序差）。

判据（跑前锁死）：
  J1 因果 = 编译版同形状探针 maxdiff 必须 = 0（fp32 也允许 ≤1e-5，超则判泄漏查因）。
  J2 公平 = compile 同等发给 TF；结论只看 OPT5+comp vs TF+comp。
用法: PYTHONPATH=/data/dynfw python benchmarks/bench_fair4.py --Ts 1024,4096,8192,16384,32768 --batch 2
"""
import sys, time, argparse
import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

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


def build_opt5():
    return to_opt5(build_base(), True, False)


def caus_fixedlen(model, T=1024, positions=(13, 519, 1020), seed=0):
    """同形状探针：只改 POS 之后的 token，POS 处 logits 必须不变。"""
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
            d = (l1 - l2).abs().max().item()
            worst = max(worst, d)
            sc = max(sc, l1.abs().max().item())
    return worst, sc


def caus_len(T, POS, seed=0):
    """长度依赖法（旧探针）——仅对未编译模型作交叉验证。"""
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T + 2,), device='cuda')
    return seg


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', default='1024,4096,8192,16384,32768')
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    print('=' * 96)
    print('J1 同形状因果探针（只改 POS 之后的 token，长度不变）')
    print('=' * 96)
    for name, mk, comp in [('TF', build_tf, False), ('TF+comp', build_tf, True),
                           ('OPT5', build_opt5, False), ('OPT5+comp', build_opt5, True)]:
        m = mk()
        if comp:
            m = torch.compile(m)
        m = m.cuda()
        try:
            for T in (1024, 4096):
                md, sc = caus_fixedlen(m, T)
                print(f'  {name:<11} T={T:<6} maxdiff={md:.6e}  (幅值 {sc:.3f})  '
                      f'{"PASS" if md < 1e-5 else "CHECK-CAUSE"}')
        except Exception as e:
            print(f'  {name:<11} ERROR {type(e).__name__}: {str(e)[:60]}')
        del m; torch.cuda.empty_cache()
    print()

    print('=' * 96)
    print(f'J2 公平桌 v4（batch={a.batch}）')
    print('=' * 96)
    COLS = [('TF', build_tf, False), ('TF+comp', build_tf, True),
            ('OPT5', build_opt5, False), ('OPT5+comp', build_opt5, True)]
    print(f"{'T':>7}" + ''.join(f'{n:>14}' for n, _, _ in COLS))
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
            row += f"{(f'{dt*1e3:.1f}ms' if dt else '--'):>14}"
            del m; torch.cuda.empty_cache()
        print(row)
    print()
    print('判据（>1 = 我们更快）:')
    for T in Ts:
        tfc = res.get(('TF+comp', T))
        o5c = res.get(('OPT5+comp', T))
        tf = res.get(('TF', T))
        s = f'  T={T:<6}'
        if tf and o5c:
            s += f' vs TF      = {tf/o5c:5.2f}x'
        if tfc and o5c:
            s += f'   vs TF+comp = {tfc/o5c:5.2f}x  <<< 公平判据'
        print(s)


if __name__ == '__main__':
    main()
