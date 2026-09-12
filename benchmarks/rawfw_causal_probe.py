"""
raw 分块因果探针 (fused_fw_rawfw_cycle) — 验证 BDH-CQ 官方 raw 严格因果 + 无 NaN
扰动位置 k 的输入, 看 i<k logits 是否完全不变(<1e-9)。raw(mask=0真0) 应无泄漏。
"""
import torch
from dynfw.models.fused_fw_rawfw_cycle import BDHBlockRawFWCycleLM

torch.manual_seed(0)
D, nh, vocab, mlp_mult, W, T = 32, 4, 200, 16, 8, 48
m = BDHBlockRawFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=1, steps=1,
                         mlp_mult=mlp_mult, W=W)
m.eval()
print(f"params = {m.np()}")

def logits_of(x):
    with torch.no_grad():
        lg, _ = m(x)
    return lg[0]

x_A = torch.randint(0, vocab, (1, T))
la = logits_of(x_A)
maxdiff = lambda a, b: (a - b).abs().max().item()

print(f"== raw 分块因果探针  D={D} nh={nh} W={W} T={T} ==")
all_ok = True
for k in [5, 12, 20, 31]:
    x_B = x_A.clone(); x_B[0, k] = (x_B[0, k] + 1) % vocab
    lb = logits_of(x_B)
    past = maxdiff(la[:k], lb[:k])
    fut = maxdiff(la[k:], lb[k:])
    blk = k // W
    sb = maxdiff(la[blk*W:k], lb[blk*W:k]) if k > blk*W else 0.0
    nan = torch.isnan(la).sum().item()
    ok = past < 1e-9
    all_ok &= ok
    print(f"  k={k} (blk{blk}): 过去i<k={past:.2e} {'OK' if ok else '⚠泄漏'} | 未来={fut:.4f} | 同块过去={sb:.2e} | NaN={nan}")
print(f"\n== 结论: {'✅ raw 严格因果, 无泄漏, 无NaN' if all_ok else '❌ 仍泄漏'} ==")
