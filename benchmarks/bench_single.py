"""单形态独立进程测速 —— 彻底消除「形态顺序污染」（实测可差 1.8×）。

用法：
    python benchmarks/bench_single.py --form tf|a_comp|a_eager|b_tl --Ts 1024,8192
一次只测一个形态（进程干净，无同类预热，也无其他形态残留）。

口径与 bench_raw_vs_tf.py / bench_tl_end2end.py 完全一致：
    VOCAB=50257 D=256 NH=8 NL=6 W=256 B=2；autocast bf16；训练 step(fwd+bwd+AdamW)；
    warmup3 / rounds3 / iters3 取中位
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
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw, to_tl_raw  # noqa: E402
from dynfw.models.transformer import TF_sdpa                          # noqa: E402

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def build_base(read_mode='raw', seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1,
                             mlp_mult=4, W=W, read_mode=read_mode)


FORMS = {
    'tf': lambda: torch.compile(TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)),
    'a_comp': lambda: torch.compile(to_opt5_raw(build_base('raw'), True, False)),
    'a_eager': lambda: to_opt5_raw(build_base('raw'), strict_bf16=True, bf16_prefix=False),
    'b_tl': lambda: to_tl_raw(build_base('raw')),
}


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def bench(m, T, B=2, rounds=3, iters=3, warmup=3, preheat=0):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(preheat):                 # GPU 预热（排除降频/冷启动假象）
        run_step(m, opt, x, y)
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


ap = argparse.ArgumentParser()
ap.add_argument('--form', required=True, choices=list(FORMS))
ap.add_argument('--Ts', type=str, default='1024,8192,16384,32768')
ap.add_argument('--warmup', type=int, default=3)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--preheat', type=int, default=0, help='测前额外预热步数（排除 GPU 低频假象）')
ap.add_argument('--out', type=str, default=None)
a = ap.parse_args()

res = {}
for T in [int(t) for t in a.Ts.split(',')]:
    try:
        m = FORMS[a.form]().cuda()
        ms = bench(m, T, rounds=a.rounds, warmup=a.warmup, preheat=a.preheat)
        res[T] = ms
        print(f'[SINGLE {a.form}] T={T:6d}  {ms*1000:9.2f} ms/step  ({2*T/ms:10.0f} tok/s)', flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        res[T] = None
        print(f'[SINGLE {a.form}] T={T:6d}  FAILED: {type(e).__name__}: {str(e)[:120]}', flush=True)

if a.out:
    json.dump({'form': a.form, 'ms_per_step': res}, open(a.out, 'w'), indent=2)
    print('saved', a.out)
