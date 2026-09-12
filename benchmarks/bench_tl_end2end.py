"""B 线端到端：TileLang raw 实现接进完整 v6 模型后的墙钟账。

口径与 A 线/09-11 公平桌完全一致（保持可比）：
    VOCAB=50257 D=256 NH=8 NL=6 W=256 B=2；autocast bf16；训练 step(前向+反向+AdamW)
    warmup3 / rounds3 / iters3 取中位

形态：
    TF+comp          TF_sdpa + compile                      （对手）
    v6raw-o5+comp    to_opt5_raw + compile                  （A 线交付）
    v6raw-TL         to_tl_raw, eager                       （B 线：TileLang 块内）
    v6raw-TL+comp    to_tl_raw + compile                    （B 线 + compile，可能因图断裂无效）

⚠️ TileLang kernel 的 S 为编译期常量 → 每个新 T 首次调用会编译（约 6s），
   由 warmup 吸收，不计入测速。

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_tl_end2end.py [--Ts 1024,8192] [--out x.json]
"""
import argparse
import json
import statistics
import sys
import time

import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM                    # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw, to_tl_raw          # noqa: E402
from dynfw.models.transformer import TF_sdpa                                    # noqa: E402

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
    ap.add_argument('--out', type=str, default='/data/dynfw/results/bench_tl_end2end.json')
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(',')]

    # ⚠️ 形态顺序必须固定：实测同一实现因进程内测量顺序不同可差 1.8×（A 线 0.88× vs 0.50×）。
    #    基线脚本 bench_raw_vs_tf.py 的顺序里，A 线之前有 eager 档先跑（预热同构路径）→ 此处对齐。
    makers = {
        'TF+comp': lambda: torch.compile(TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)),
        'v6raw-o5': lambda: to_opt5_raw(build_base('raw'), strict_bf16=True, bf16_prefix=False),
        'v6raw-o5+comp': lambda: torch.compile(to_opt5_raw(build_base('raw'), True, False)),
        'v6raw-TL': lambda: to_tl_raw(build_base('raw')),
        'v6raw-TL+comp': lambda: torch.compile(to_tl_raw(build_base('raw'))),
    }

    res = {}
    for name, mk in makers.items():
        res[name] = {}
        for T in Ts:
            try:
                m = mk().cuda()
                ms = bench_median(m, T)
                res[name][T] = ms
                print(f'{name:18s} T={T:6d}  {ms*1000:9.2f} ms/step  ({2*T/ms:10.0f} tok/s)', flush=True)
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                res[name][T] = None
                print(f'{name:18s} T={T:6d}  FAILED: {type(e).__name__}: {str(e)[:100]}', flush=True)
                torch.cuda.empty_cache()

    tf = res.get('TF+comp', {})
    print('\n=== 相对 TF+comp 的加速比（>1 = 我们快）===')
    print(f"{'T':>8s} {'v6raw-o5+comp(A线)':>20s} {'v6raw-TL(B线)':>16s} {'v6raw-TL+comp':>16s}")
    for T in Ts:
        t = tf.get(T)

        def sp(k):
            v = res.get(k, {}).get(T)
            return f'{t/v:.2f}x' if (t and v) else 'n/a'
        print(f'{T:8d} {sp("v6raw-o5+comp"):>20s} {sp("v6raw-TL"):>16s} {sp("v6raw-TL+comp"):>16s}')

    print('\n=== 各形态相对 A 线（v6raw-o5+comp）===')
    for T in Ts:
        a = res.get('v6raw-o5+comp', {}).get(T)
        if not a:
            continue
        row = [f'T={T:6d}', f'A={a*1000:7.2f}ms']
        for k in ('v6raw-TL', 'v6raw-TL+comp', 'v6raw-o5'):
            v = res.get(k, {}).get(T)
            row.append(f'{k}={a/v:.3f}x' if v else f'{k}=n/a')
        print('  '.join(row))

    print('\n=== B 线相对 A 线（同语义、同口径）===')
    for T in Ts:
        a = res.get('v6raw-o5+comp', {}).get(T)
        c = res.get('v6raw-TL', {}).get(T)
        if a and c:
            print(f'T={T:6d}  A={a*1000:8.2f} ms  B={c*1000:8.2f} ms  →  B/A 提速 {a/c:.3f}x')

    json.dump({'config': {'VOCAB': VOCAB, 'D': D, 'NH': NH, 'NL': NL, 'W': W, 'B': 2,
                          'dtype': 'autocast bf16', 'mode': 'train step (fwd+bwd+AdamW)'},
               'ms_per_step': res}, open(args.out, 'w'), indent=2)
    print(f'\n结果已存 {args.out}')


if __name__ == '__main__':
    main()
