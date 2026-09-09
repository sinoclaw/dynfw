"""
逐位置可视化 FWAttention 泄漏: 扰动 V[k], 看每个输出位置 i 的 diff 分布
定位: 受影响位置是"仅未来"还是"整块/整段" — 决定泄漏机制
"""
import torch
from dynfw.models.fused_fw_fw_cycle import FWAttention, Config

torch.manual_seed(1)
D, nh, W, T = 32, 4, 8, 48
cfg = Config(1, D, nh, 16, 200)
N = cfg.mlp_internal_dim_multiplier * D // nh
attn = FWAttention(cfg); attn.eval()

torch.manual_seed(2)
B = 1
Q = torch.randn(B, nh, T, N)
V = torch.randn(B, 1, T, D)

def fwd(Q, K, V_):
    with torch.no_grad():
        return attn(Q, K, V_, W=W)[0]

out_A = fwd(Q, Q, V)

# 扰动 V 位置 k, 看每个输出位置 i 的 (最大diff, 属块)
for k in [12, 24]:
    V_B = V.clone(); V_B[0, 0, k] += 1.0
    out_B = fwd(Q, Q, V_B)
    diff = (out_A - out_B).abs().amax(dim=(0, 1, 3))  # [T]
    print(f"=== 扰动 V[k={k}] (k块={k//W}) 逐位置diff ===")
    for i in range(T):
        blk = i // W
        flag = "⚠泄漏" if diff[i].item() > 1e-9 else "   "
        print(f"   pos{i:2d} (blk{blk}): {diff[i].item():.3e}  {flag}")
    # 摘要: 受影响的最小/最大位置
    affected = [i for i in range(T) if diff[i].item() > 1e-9]
    print(f"   → 受影响位置范围: [{min(affected) if affected else 'none'}, {max(affected) if affected else 'none'}]")
    print(f"   → 受影响块: {sorted(set(i//W for i in affected)) if affected else 'none'}")
    print()
