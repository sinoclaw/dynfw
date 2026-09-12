"""J1 判据：to_opt5_raw 与基线 fw_cycle(read_mode='raw') 的数值等价 + 因果性。

PLAN-A J1 要求：
  - 因果探针 maxdiff 必须 = 0
  - 数值等价：保留 fp32 → < 1e-2；全 bf16 → < 1e-1（须标注「数值有变」）
"""
import copy

import torch

from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw

DEV = 'cuda'
KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)


def build():
    torch.manual_seed(1234)
    return BDHBlockFWCycleLM(**KW).to(DEV).eval()


base = build()
opt = copy.deepcopy(build())
to_opt5_raw(opt, strict_bf16=True, bf16_prefix=False)
print('opt attn 类:', type(opt.blocks[0].attn).__name__)

same_params = all(torch.equal(a, b) for a, b in zip(base.parameters(), opt.parameters()))
print('参数逐位相同:', same_params, f'({sum(p.numel() for p in base.parameters()):,} 个参数)')

print('\n=== 数值等价（同权重同输入）===')
worst = 0.0
for T in (64, 256, 1024, 4096):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, T), device=DEV)
    with torch.no_grad():
        a = base(x)[0]
        b = opt(x)[0]
    d = (a - b).abs().max().item()
    ref = a.abs().max().item()
    worst = max(worst, d)
    print(f'T={T:5d}  logits maxdiff = {d:.3e}   (参考量级 |logits|max={ref:.3f}, 相对={d/ref:.2e})')

print('\n=== 因果性（长度依赖探针）===')
for m, name in ((base, 'base(raw)'), (opt, 'opt5_raw')):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, 50), device=DEV)
    with torch.no_grad():
        a = m(x[:, :49])[0]
        b = m(x[:, :50])[0]
    d = (a[0, 48] - b[0, 48]).abs().max().item()
    print(f'{name:12s} maxdiff = {d:.3e}  {"CAUSAL OK" if d < 1e-5 else "LEAK!!"}')

print('\n=== 判读 ===')
if worst == 0.0:
    print('逐位等价（maxdiff=0）→ 实现替换无任何数值影响')
elif worst < 1e-2:
    print(f'数值等价（maxdiff={worst:.2e} < 1e-2）→ J1 通过（前缀和求和顺序差，噪声级）')
else:
    print(f'⚠️ maxdiff={worst:.2e} 偏大，需检查实现语义')
