"""定位「单进程 a_comp 15.98ms vs 混合进程 8.72ms」的 1.8× 差异来源。

四个变体（都在独立进程里跑，各建独立模型；关键区别是 compile 前进程里干过什么）：
  V1 直接 compile(opt5_raw)                        ← 单进程现状
  V2 compile 前先用【同一实例】eager 跑 5 步        ← 填 mask 缓存 + 热身该实例
  V3 compile 前先跑【另一个 opt5_raw 实例】5 步      ← 只热身"环境"，不动本实例
  V4 compile 前先跑【TF 形态】5 步                  ← 排除"任意 GPU 负载"的效果

若 V3 就变快 ⇒ 差异来自【进程级环境状态】（显存池/库缓存），而非本实例的 mask 缓存。
"""
import sys
import time

import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM            # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw              # noqa: E402
from dynfw.models.transformer import TF_sdpa                            # noqa: E402

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
variant = sys.argv[1] if len(sys.argv) > 1 else 'V1'
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024


def build():
    torch.manual_seed(0)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1,
                             mlp_mult=4, W=W, read_mode='raw')


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def bench(m, T, B=2, rounds=3, iters=3, warmup=3):
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


x = torch.randint(0, VOCAB, (2, T), device='cuda')
y = torch.randint(0, VOCAB, (2, T), device='cuda')

if variant == 'V3':
    other = to_opt5_raw(build(), True, False).cuda()
    o = torch.optim.AdamW(other.parameters(), lr=1e-4)
    for _ in range(5):
        run_step(other, o, x, y)
    torch.cuda.synchronize()
    del other, o
    torch.cuda.empty_cache()
elif variant == 'V4':
    other = torch.compile(TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=262144)).cuda()
    o = torch.optim.AdamW(other.parameters(), lr=1e-4)
    for _ in range(5):
        run_step(other, o, x, y)
    torch.cuda.synchronize()
    del other, o
    torch.cuda.empty_cache()

m = to_opt5_raw(build(), True, False).cuda()
if variant == 'V2':
    o2 = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(5):
        run_step(m, o2, x, y)
    torch.cuda.synchronize()
    del o2

mc = torch.compile(m)
t = bench(mc, T)
print(f'[{variant}] T={T}  {t * 1000:8.2f} ms/step')
