"""验证：`_mask_cache` 类变量在 torch.compile trace 时被固化，导致 1.8× 差异。

三个变体（同一进程内，各用独立模型实例）：
  V1 直接 compile                          （单进程现状，实测 15.98ms@1024）
  V2 先 eager 跑 1 次（填好 _mask_cache）再 compile
  V3 compile，但 mask 在 compile 前已建好（手动预热缓存）

若 V2/V3 ≈ 8.7ms 而 V1 ≈ 16ms ⇒ 机制成立：compile 应在 mask 缓存就绪后进行。
"""
import sys
import time

import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM                        # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw, FWAttentionOpt5Raw       # noqa: E402

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256


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


T = 1024
print(f'=== T={T}，三个变体（同进程）===')
print(f'（入口: _mask_cache 状态 = {FWAttentionOpt5Raw._mask_cache}）')

# V1: 直接 compile（缓存为空）
FWAttentionOpt5Raw._mask_cache = None
m1 = build().cuda()
dummy = torch.optim.AdamW(m1.parameters(), lr=1e-4)
x = torch.randint(0, VOCAB, (2, T), device='cuda')
y = torch.randint(0, VOCAB, (2, T), device='cuda')
m1c = torch.compile(m1)
t1 = bench(m1c, T)
print(f'V1 直接 compile                    {t1 * 1000:8.2f} ms   '
      f'(mask_cache={"已建" if FWAttentionOpt5Raw._mask_cache else "空"})')

# V2: 先 eager 跑一次（建缓存）再 compile
FWAttentionOpt5Raw._mask_cache = None
m2 = build().cuda()
with torch.autocast('cuda', dtype=torch.bfloat16):
    m2(x, y)                                     # eager 一次 → 建 mask 缓存
print(f'    eager 预热后 mask_cache = {"已建" if FWAttentionOpt5Raw._mask_cache else "空"}')
m2c = torch.compile(m2)
t2 = bench(m2c, T)
print(f'V2 先 eager 1 次再 compile         {t2 * 1000:8.2f} ms')

# V3: 手动预热 mask 缓存（不跑 eager），再 compile
FWAttentionOpt5Raw._mask_cache = None
FWAttentionOpt5Raw._tril_mask(W, torch.device('cuda'))    # 直接建缓存
m3 = build().cuda()
m3c = torch.compile(m3)
t3 = bench(m3c, T)
print(f'V3 仅预建 mask 缓存再 compile      {t3 * 1000:8.2f} ms')

print()
print('判读：若 V2/V3 明显快于 V1（接近 8.7ms）⇒ mask 缓存在 trace 时被固化的假设成立；')
print('      若三者一致 ⇒ 差异来自其他进程内状态。')
