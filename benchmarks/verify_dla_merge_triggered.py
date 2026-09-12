"""DLA 槽机制「是否被真正触发」的正确验证。
关键：chunk 数 = T/W。T=64,W=64 → 1 chunk（槽永远不满，任何 K 都等价于 FW）。
      T=256,W=64 → 4 chunks，K<=4 时才触发信息感知合并 → DLA 才与 FW 分道。
"""
import torch

from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
from dynfw.models.fused_fw_dla_topk_cycle import BDHBlockSlotCycleLM
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM

KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)


def build(fn):
    torch.manual_seed(1234)
    return fn()


fw = build(lambda: BDHBlockFWCycleLM(**KW))
d8 = build(lambda: BDHBlockDLACycleLM(K=8, **KW))
d4 = build(lambda: BDHBlockDLACycleLM(K=4, **KW))
d2 = build(lambda: BDHBlockDLACycleLM(K=2, **KW))

for T in (64, 256, 1024):
    nchunk = T // 64
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, T))
    with torch.no_grad():
        lf = fw(x)[0]
        a8 = (lf - d8(x)[0]).abs().max().item()
        a4 = (lf - d4(x)[0]).abs().max().item()
        a2 = (lf - d2(x)[0]).abs().max().item()
    print(f'T={T:5d} (chunk数={nchunk:2d})  fw~dla(K=8)={a8:.3e}  fw~dla(K=4)={a4:.3e}  fw~dla(K=2)={a2:.3e}')
    print(f'{"":22s}判定: K=8 {"等价(未触发)" if a8 < 1e-6 else "已分化"} | '
          f'K=4 {"等价" if a4 < 1e-6 else "已分化"} | K=2 {"等价" if a2 < 1e-6 else "已分化"}')

# v8 slot_topk 同样检查（read_mode=softmaxK, topk=2）
s8 = build(lambda: BDHBlockSlotCycleLM(K=8, **KW, read_mode='softmaxK', topk=2))
s2 = build(lambda: BDHBlockSlotCycleLM(K=2, **KW, read_mode='softmaxK', topk=2))
print()
for T in (256, 1024):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, T))
    with torch.no_grad():
        lf = fw(x)[0]
        b8 = (lf - s8(x)[0]).abs().max().item()
        b2 = (lf - s2(x)[0]).abs().max().item()
    print(f'[v8 slot_topk] T={T:5d}  fw~slot(K=8)={b8:.3e}  fw~slot(K=2)={b2:.3e}')
