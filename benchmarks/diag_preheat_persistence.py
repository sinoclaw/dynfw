"""验证：eager 预热带来的 1.87× 是否【持久】（而非只快前几步）。

流程：compile → 逐轮记录 → 中间插入 eager 预热 → 继续记录
判读：若预热后长时间稳定在 ~8.7ms ⇒ 持久，可作为生产口径；
      若只是短暂变快又回落到 16ms ⇒ 不是真实稳态，不能报。
"""
import sys
import time

import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM          # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw            # noqa: E402

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60


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


x = torch.randint(0, VOCAB, (2, T), device='cuda')
y = torch.randint(0, VOCAB, (2, T), device='cuda')

m = to_opt5_raw(build(), True, False).cuda()
o = torch.optim.AdamW(m.parameters(), lr=1e-4)
mc = torch.compile(m)

# 阶段 1：compile 后直接跑（不预热）
for _ in range(3):
    run_step(mc, o, x, y)
torch.cuda.synchronize()
phase1 = []
for _ in range(5):
    t0 = time.time()
    for _ in range(3):
        run_step(mc, o, x, y)
    torch.cuda.synchronize()
    phase1.append((time.time() - t0) / 3)
print(f'阶段1 compile 直接跑（无 eager 预热）: {[f"{v*1000:.2f}" for v in phase1]} ms')

# 插入 eager 预热（用另一个实例走 eager 路径）
print('>>> 插入 eager 预热（另一个 opt5_raw 实例跑 5 步）...')
other = to_opt5_raw(build(), True, False).cuda()
oo = torch.optim.AdamW(other.parameters(), lr=1e-4)
for _ in range(5):
    run_step(other, oo, x, y)
torch.cuda.synchronize()
del other, oo
torch.cuda.empty_cache()

# 阶段 2：再次测 compile 版
phase2 = []
for _ in range(5):
    t0 = time.time()
    for _ in range(3):
        run_step(mc, o, x, y)
    torch.cuda.synchronize()
    phase2.append((time.time() - t0) / 3)
print(f'阶段2 预热后 compile:              {[f"{v*1000:.2f}" for v in phase2]} ms')

# 阶段 3：长跑 60 步，看是否漂移
print(f'>>> 长跑 {STEPS} 步（每 15 步报一次）...')
m.mark = None
t_all = time.time()
for i in range(STEPS):
    run_step(mc, o, x, y)
    if (i + 1) % 15 == 0:
        torch.cuda.synchronize()
        el = (time.time() - t_all) / 15
        print(f'    step {i+1:4d}: {el*1000:8.2f} ms/step')
        t_all = time.time()

print(f'\n判读: 阶段1={sum(phase1)/len(phase1)*1000:.2f}ms  阶段2={sum(phase2)/len(phase2)*1000:.2f}ms')
