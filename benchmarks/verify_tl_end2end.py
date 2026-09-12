"""B 线 J1'：TileLang 版接进模型后的端到端数值一致性 + 因果性。

对比三方（同权重、同输入、同 dtype）：
    base  = BDHBlockFWCycleLM(read_mode='raw')        （基线，未优化）
    o5    = to_opt5_raw(base 副本)                     （A 线交付）
    tl    = to_tl_raw(base 副本)                       （B 线：TileLang 块内）

判据：tl 相对 base/o5 的 logits maxdiff 应在 bf16 噪声级；因果探针 maxdiff = 0。
"""
import copy

import torch

from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw, to_tl_raw

DEV = 'cuda'
KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)


def build():
    torch.manual_seed(1234)
    return BDHBlockFWCycleLM(**KW).to(DEV).eval()


m_base = build()
m_o5 = copy.deepcopy(m_base)
to_opt5_raw(m_o5, strict_bf16=True, bf16_prefix=False)
m_tl = copy.deepcopy(m_base)
to_tl_raw(m_tl)

print('attn 类: base=%s  o5=%s  tl=%s' % (
    type(m_base.blocks[0].attn).__name__, type(m_o5.blocks[0].attn).__name__,
    type(m_tl.blocks[0].attn).__name__))
ps = [sum(p.numel() for p in m.parameters()) for m in (m_base, m_o5, m_tl)]
print('参数量: base=%s o5=%s tl=%s  全等=%s' % (*[f'{p:,}' for p in ps], len(set(ps)) == 1))

print('\n=== 端到端 logits 一致性（autocast bf16）===')
for T in (256, 1024, 4096):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, T), device=DEV)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        lb = m_base(x)[0].float()
        lo = m_o5(x)[0].float()
        lt = m_tl(x)[0].float()
    ref = lb.abs().max().item()
    print(f'T={T:5d}  |logits|max={ref:.4f}   tl-vs-base={((lt-lb).abs().max().item()):.3e}   '
          f'tl-vs-o5={((lt-lo).abs().max().item()):.3e}   (相对: {((lt-lb).abs().max().item())/ref:.2e})')

print('\n=== 因果性（长度依赖探针）===')
for m, name in ((m_base, 'base'), (m_o5, 'o5'), (m_tl, 'tl')):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, 200), device=DEV)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = m(x[:, :199])[0]
        b = m(x[:, :200])[0]
    d = (a[0, 198].float() - b[0, 198].float()).abs().max().item()
    print(f'{name:5s} maxdiff = {d:.3e}  {"CAUSAL OK" if d < 1e-3 else "LEAK!!"}')
