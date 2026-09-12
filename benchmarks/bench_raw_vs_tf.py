"""A 线 J2：v6 **raw** 语义下的墙钟账（公平桌：双方同优化等级 = 都 compile）。

跑前锁死的判据（沿用 PLAN-A J2，事后不改）：
    raw-opt 相对 TF 加速比 ≥1.5×   → ✅ 卖点成立
    0.8 ~ 1.5×                    → ⚠️ 打平，可再迭代
    < 0.8×                        → ❌ 卖点不成立，如实报

口径（与 09-11 方案A 公平桌完全一致，保持可比）：
    VOCAB=50257, D=256, NH=8, NL=6, W=256;  B=2;  autocast bf16;  训练 step(前向+反向+AdamW)
    T ∈ {1024, 8192, 16384, 32768}；warmup 3 / rounds 3 / iters 3，取中位

对照四方：
    TF+comp         TF_sdpa + torch.compile                （对手，O(T²) 精确 + SDPA 融合核）
    v6soft-o5+comp  to_opt5(softmax) + compile            （v6 旧语义参考，即 09-11 的交付形态）
    v6raw-o5         to_opt5_raw, eager                    （看 compile 贡献）
    v6raw-o5+comp   to_opt5_raw + compile                 （**主对象：v6 新语义最优形态**）

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_raw_vs_tf.py [--out xx.json] [--Ts 1024,8192]
"""
import argparse
import json
import sys
import time

import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM          # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5, to_opt5_raw  # noqa: E402
from dynfw.models.transformer import TF_sdpa                          # noqa: E402

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(read_mode='raw', seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1,
                             mlp_mult=4, W=W, read_mode=read_mode)


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


def bench_median(m, T, B=2, rounds=3, iters=3, warmup=3):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(warmup):
        run_step(m, opt, x, y)
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.time()
        for _ in range(iters):
            run_step(m, opt, x, y)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ts', type=str, default='1024,8192,16384,32768')
    ap.add_argument('--out', type=str, default='/data/dynfw/results/bench_raw_vs_tf.json')
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(',')]

    makers = {
        'TF+comp': lambda: torch.compile(TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)),
        'v6soft-o5+comp': lambda: torch.compile(to_opt5(build_base('softmax'), True, False, False)),
        'v6raw-o5': lambda: to_opt5_raw(build_base('raw'), strict_bf16=True, bf16_prefix=False),
        'v6raw-o5+comp': lambda: torch.compile(to_opt5_raw(build_base('raw'), True, False)),
    }

    res = {}
    for name, mk in makers.items():
        res[name] = {}
        for T in Ts:
            try:
                m = mk().cuda()
                ms = bench_median(m, T)
                res[name][T] = ms
                print(f'{name:18s} T={T:6d}  {ms*1000:9.2f} ms/step  '
                      f'({2*T/ms:10.0f} tok/s)', flush=True)
                del m
                torch.cuda.empty_cache()
            except RuntimeError as e:
                res[name][T] = None
                print(f'{name:18s} T={T:6d}  FAILED: {str(e)[:110]}', flush=True)
                torch.cuda.empty_cache()

    print('\n=== 加速比（相对 TF+comp；>1 = 我们快）===')
    print(f"{'T':>8s} {'v6raw-o5+comp':>16s} {'v6soft-o5+comp':>16s} {'v6raw-o5(eager)':>18s}")
    for T in Ts:
        tf = res['TF+comp'].get(T)
        def sp(k):
            v = res.get(k, {}).get(T)
            return f'{tf/v:.2f}x' if (tf and v) else 'n/a'
        print(f'{T:8d} {sp("v6raw-o5+comp"):>16s} {sp("v6soft-o5+comp"):>16s} {sp("v6raw-o5"):>18s}')

    print('\n=== 判据对照（raw-opt+comp vs TF+comp）===')
    for T in Ts:
        tf = res['TF+comp'].get(T)
        v = res['v6raw-o5+comp'].get(T)
        if not (tf and v):
            continue
        r = tf / v
        verdict = '✅ 卖点成立' if r >= 1.5 else ('⚠️ 打平/可再迭代' if r >= 0.8 else '❌ 卖点不成立')
        print(f'T={T:6d}  加速比 {r:.3f}x  → {verdict}')

    json.dump({'config': {'VOCAB': VOCAB, 'D': D, 'NH': NH, 'NL': NL, 'W': W, 'B': 2,
                          'dtype': 'autocast bf16', 'mode': 'train step (fwd+bwd+AdamW)'},
               'ms_per_step': res}, open(args.out, 'w'), indent=2)
    print(f'\n结果已存 {args.out}')


if __name__ == '__main__':
    main()
