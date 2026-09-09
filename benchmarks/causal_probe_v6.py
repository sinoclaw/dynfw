"""
因果性扰动探针 (v6 BDHBlockFWCycleLM) — L2 判据层
原理: 严格因果 => 改变位置 k 的输入 token, 位置 i<k 的输出 logits 必须完全不变。
   (a) 块内: 位置 i 只看 <i (tril diagonal=-1)
   (b) 跨块: fast-weight 检索用"更新前"历史, 不含当前块
测 W=8, T=48 (3 块), 扰动多个位置 k, 检查 i<k 是否 diff==0 (严格因果) / i>k 是否改变。
"""
import torch
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM

torch.manual_seed(0)
D, nh, vocab, mlp_mult, W, T = 32, 4, 200, 16, 8, 48
m = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=1, steps=1,
                      mlp_mult=mlp_mult, W=W)
m.eval()  # eval: 无 dropout 影响, 确定性前向

def logits_of(x):
    with torch.no_grad():
        lg, _ = m(x)  # forward(x, None) -> (logits, None)
    return lg[0]      # [T, vocab]

x_A = torch.randint(0, vocab, (1, T))
la = logits_of(x_A)

def maxdiff(a, b):
    return (a - b).abs().max().item()

print(f"== v6 因果性扰动探针  D={D} nh={nh} W={W} T={T} vocab={vocab} ==")
print(f"params = {m.np()}")
print()
# 逐位置报告: 扰动 k, 看每个输出位置 i 的 diff
for k in [5, 12, 20, 31]:
    x_B = x_A.clone()
    x_B[0, k] = (x_B[0, k] + 1) % vocab
    lb = logits_of(x_B)
    # 分块位置
    blk = k // W
    print(f"--- 扰动位置 k={k} (块 blk={blk}, 块内偏移 r={k % W}) ---")
    # 检查严格因果: 所有 i<k 应 diff==0
    past_diffs = [maxdiff(la[i], lb[i]) for i in range(k)]
    future_diffs = [maxdiff(la[i], lb[i]) for i in range(k, T)]
    print(f"  过去 i<k (严格因果要求==0): max={max(past_diffs):.2e}  "
          f"{'OK' if max(past_diffs) < 1e-9 else '⚠ 泄漏!'}")
    print(f"  未来 i>=k (应变化): max={max(future_diffs):.3f}")
    # 同块内过去位置 (块内因果): 位置 i in [blk*W, k) 应仍==0
    same_blk_past = [maxdiff(la[i], lb[i]) for i in range(blk * W, k)]
    print(f"  同块内过去 i in [{blk*W},{k}): max={max(same_blk_past):.2e} "
          f"{'OK' if (not same_blk_past or max(same_blk_past) < 1e-9) else '⚠ 块内泄漏!'}")
    print()

print("== 结论 ==")
print("若所有扰动下 过去 i<k 的 diff 均 < 1e-9 → v6 严格因果 (块内 causal + 跨块 fast-weight 检索无泄漏)")
print("若出现泄漏 → 需查 fast-weight 更新是否混入未来位置")
