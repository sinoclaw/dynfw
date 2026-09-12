"""判定 v6 fw_cycle 与 v7 dla_cycle 在 W=64 下是否数值等价（同 seed 同参初始化）。
若逐位相同 → 说明 DLA 的槽机制在该配置下未触发差异化（读侧同为无差别 sum），
则矩阵里 "dla 93.05" 实为 fw 的值，DLA 的差异化必须用更小 K（触发合并）才显形。
"""
import torch

from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM

KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)

torch.manual_seed(1234)
a = BDHBlockFWCycleLM(**KW)
torch.manual_seed(1234)
b = BDHBlockDLACycleLM(K=8, **KW)
torch.manual_seed(1234)
b16 = BDHBlockDLACycleLM(K=16, **KW)
torch.manual_seed(1234)
b4 = BDHBlockDLACycleLM(K=4, **KW)   # K < chunk 数（4 chunk）→ 应触发合并

print(f'参数量: fw={a.np():,}  dla(K=8)={b.np():,}  dla(K=16)={b16.np():,}  dla(K=4)={b4.np():,}')

sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
print('参数名集合一致:', set(sa) == set(sb))
if set(sa) == set(sb):
    diffnames = [k for k in sa if not torch.equal(sa[k], sb[k])]
    print('同 seed 下参数逐位相同的张量数:', len(sa) - len(diffnames), '/', len(sa))
    if diffnames:
        print('  差异参数(前5):', diffnames[:5])

torch.manual_seed(7)
x = torch.randint(0, 151936, (1, 64))
with torch.no_grad():
    la = a(x)[0]
    lb = b(x)[0]
    lb16 = b16(x)[0]
    lb4 = b4(x)[0]

print(f'\nlogits 对比（同 seed 同输入 T=64）:')
print(f'  fw      vs dla(K=8)  maxdiff = {(la - lb).abs().max().item():.6e}')
print(f'  fw      vs dla(K=16) maxdiff = {(la - lb16).abs().max().item():.6e}')
print(f'  fw      vs dla(K=4)  maxdiff = {(la - lb4).abs().max().item():.6e}')
print(f'  dla(K=8) vs dla(K=4) maxdiff = {(lb - lb4).abs().max().item():.6e}')
