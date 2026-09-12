"""速度账补测：**同一 KL 口径**下三方并列。

  A) v6 opt5_raw, eager
  B) v6 opt5_raw, +torch.compile（编译前先 eager 预热 —— 今天定位的 1.87× 机制）
  C) tf SDPA, +torch.compile（公平性铁律：要上 compile 就双方都上）

口径：完整训练 step = forward_hidden + chunked_kl_loss(分块, chunk=2048) + backward + AdamW。
     假教师 logits 常驻显存（隔离 mmap IO）。预热 20 / 测量 30 step 中位。
用法: python bench_step_speed_compile.py [T ...]
"""
import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM      # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw        # noqa: E402
from dynfw.models.transformer import TF_sdpa                      # noqa: E402
from dynfw.training.chunked_kl import chunked_kl_loss             # noqa: E402

VOCAB, D, NH, NL, MM, W = 151936, 128, 16, 2, 64, 64
DEV = 'cuda'
WARM, MEAS = 20, 30


def build(kind, T):
    torch.manual_seed(0)
    if kind == 'v6':
        m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=MM, W=W,
                              read_mode='raw')
        m = to_opt5_raw(m, strict_bf16=True, bf16_prefix=False)
    else:
        m = TF_sdpa(D=D, nh=NH, n_layer=NL, vocab=VOCAB, maxT=T)
    return m.to(DEV).train()


def one_step(m, opt, x, t_lg):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = m.forward_hidden(x)
        Wh, bh = m.head_params()
        loss = chunked_kl_loss(h, Wh, bh, t_lg, chunk=2048)
    opt.zero_grad(); loss.backward(); opt.step()


def bench(kind, T, use_compile):
    m = build(kind, T)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (1, T), device=DEV)
    t_lg = torch.zeros(1, T, VOCAB, dtype=torch.float16, device=DEV)

    # 编译前先跑 eager（今天 diag_order_effect 的 V3 变体：编译时机决定 1.87×）
    for _ in range(5):
        one_step(m, opt, x, t_lg)
    torch.cuda.synchronize()
    if use_compile:
        try:
            # 关键：实际调用的是 forward_hidden（不是 forward），必须编译这个方法本身，
            # 否则 compile 完全不生效（eager/comp 读数相同）。
            m.forward_hidden = torch.compile(m.forward_hidden)
        except Exception as e:
            return None, f'compile 失败: {type(e).__name__}'
        try:
            one_step(m, opt, x, t_lg)      # 触发编译
            torch.cuda.synchronize()
        except Exception as e:
            return None, f'编译执行失败: {type(e).__name__}: {str(e)[:80]}'

    for _ in range(WARM):
        one_step(m, opt, x, t_lg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(MEAS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one_step(m, opt, x, t_lg)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    del m, opt, x, t_lg
    torch.cuda.empty_cache()
    return statistics.median(ts), None


ap = argparse.ArgumentParser()
ap.add_argument('Ts', nargs='*', type=int, default=[8192, 16384, 32768])
ap.add_argument('--compile', action='store_true')
a = ap.parse_args()

print(f'{"T":>7s} {"v6_eager":>10s} {"v6_comp":>10s} {"tf_eager":>10s} {"tf_comp":>10s} {"tf/v6_c":>9s}')
print('-' * 62)
for T in a.Ts:
    ve, _ = bench('v6', T, False)
    vc, ev = bench('v6', T, a.compile)
    te, _ = bench('tf', T, False)
    tc, et = bench('tf', T, a.compile)
    r = f'{tc/vc:.3f}' if (vc and tc) else f'({ev or et})'
    s = lambda x: f'{x:8.2f}ms' if x else '      -- '
    print(f'{T:7d} {s(ve):>10s} {s(vc):>10s} {s(te):>10s} {s(tc):>10s} {r:>9s}', flush=True)
print('\n判读: tf/v6_c < 1 ⇒ TF 快; > 1 ⇒ v6(compile) 快。')
