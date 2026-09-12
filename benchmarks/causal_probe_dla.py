"""因果性扰动探针 (v7 DLA BDHBlockDLACycleLM) — L2 判据层
测修复(diag=0 + mask=-inf)后 DLA 是否严格因果, 尤其覆盖"触发合并"场景(K<块数)。
判据: 扰动位置 k 的 token, 过去 i<k 的 logits 必须完全不变(不偷看未来)。
"""
import torch
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM

torch.manual_seed(0)
D, nh, vocab, mlp_mult, W, T, K = 32, 4, 200, 16, 8, 48, 4
# W=8, T=48 -> 6 块; K=4 -> 第5块触发合并 (测合并是否泄漏)
m = BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=1, steps=1,
                       mlp_mult=mlp_mult, W=W, K=K)
m.eval()

def logits_of(x):
    with torch.no_grad():
        lg, _ = m(x)
    return lg[0]

x_A = torch.randint(0, vocab, (1, T))
la = logits_of(x_A)

def maxdiff(a, b):
    return (a - b).abs().max().item()

print(f"== v7 DLA 因果性扰动探针  D={D} nh={nh} W={W} T={T} K={K} vocab={vocab} ==")
print(f"params = {m.np()}")
print()
for k in [3, 11, 23, 33, 43]:   # 覆盖不同块 + 合并后
    x_B = x_A.clone()
    x_B[0, k] = (x_B[0, k] + 1) % vocab
    lb = logits_of(x_B)
    blk = k // W
    past = [maxdiff(la[i], lb[i]) for i in range(k)]
    future = [maxdiff(la[i], lb[i]) for i in range(k, T)]
    s = f"扰动 k={k:2d} (块{blk}, 合并后={blk>=K}): 过去 i<k max={max(past):.2e} "
    s += "OK" if max(past) < 1e-9 else "⚠泄漏!"
    s += f" | 未来 i>=k max={max(future):.3f}"
    print(s)

print("\n== 结论 ==")
print("若所有扰动 过去 i<k 均 <1e-9 → DLA 修复后严格因果(含合并触发场景)")
