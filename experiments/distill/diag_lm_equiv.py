# 诊断2: v6 BDHBlockFWCycleLM vs DLA BDHBlockDLACycleLM 完整前向对比 (生产配置 W=64, n_layer=2, K=8)
import torch, sys
sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM

torch.manual_seed(0)
m6 = BDHBlockFWCycleLM(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)
torch.manual_seed(0)
md = BDHBlockDLACycleLM(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64, K=8)
print("v6 params", m6.np(), " DLA params", md.np())

torch.manual_seed(42)
x = torch.randint(0, 151936, (2, 256))
lg6 = m6(x)[0]
lgd = md(x)[0]
d = (lg6 - lgd).abs()
print("logits shape", tuple(lg6.shape), tuple(lgd.shape))
print("logits maxdiff", d.max().item(), " meandiff", d.mean().item())
# 原始 loss (未蒸馏 raw logits, 只是对比前向是否等价)
loss6 = torch.nn.functional.cross_entropy(lg6.view(-1, 151936), x.view(-1))
lossd = torch.nn.functional.cross_entropy(lgd.view(-1, 151936), x.view(-1))
print("raw CE loss6", loss6.item(), " lossd", lossd.item())
print("=> 前向是否等价:", "是 (logits一致)" if d.max().item() < 1e-5 else "否 (有实现差异)")
