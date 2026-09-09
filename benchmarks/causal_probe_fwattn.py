"""
定位 v6 因果泄漏源 — 隔离 FWAttention 单模块因果性
若 FWAttention 本身泄漏 => 问题在注意力; 若 FWAttention 干净 => 泄漏在 block 外层。
"""
import torch
from dynfw.models.fused_fw_fw_cycle import FWAttention, Config

torch.manual_seed(1)
D, nh, W, T = 32, 4, 8, 48
cfg = Config(1, D, nh, 16, 200)
N = cfg.mlp_internal_dim_multiplier * D // nh  # =128
attn = FWAttention(cfg)
attn.eval()

freqs = attn.freqs  # [1,1,1,N]

def fwd(Q, K, V, memories=None):
    with torch.no_grad():
        return attn(Q, K, V, memories=memories, W=W)[0]  # [B,nh,T,D]

# 随机固定输入
torch.manual_seed(2)
B = 1
Q = torch.randn(B, nh, T, N)
V = torch.randn(B, 1, T, D)
out_A = fwd(Q, Q, V)  # K is Q

print(f"== FWAttention 单模块因果 (D={D} nh={nh} N={N} W={W} T={T}) ==")
for k in [5, 12, 20, 31]:
    V_B = V.clone()
    V_B[0, 0, k] += 1.0   # 扰动 V 位置 k
    out_B = fwd(Q, Q, V_B)
    past = (out_A[:, :, :k] - out_B[:, :, :k]).abs().max().item()
    fut = (out_A[:, :, k:] - out_B[:, :, k:]).abs().max().item()
    blk = k // W
    sameblk = (out_A[:, :, blk*W:k] - out_B[:, :, blk*W:k]).abs().max().item() if k > blk*W else 0.0
    print(f"  k={k} (blk={blk}): 过去i<k max={past:.2e} {'OK' if past<1e-9 else '⚠泄漏'} | "
          f"未来i>=k max={fut:.4f} | 同块过去 max={sameblk:.2e}")

# 再测: 扰动 Q(即K=x_sparse) 位置 k
print("\n== 扰动 Q 位置 k (K is Q) ==")
for k in [12, 24]:
    Q_B = Q.clone()
    Q_B[0, :, k] += 1.0
    out_B = fwd(Q_B, Q_B, V)
    past = (out_A[:, :, :k] - out_B[:, :, :k]).abs().max().item()
    print(f"  k={k}: 过去i<k max={past:.2e} {'OK' if past<1e-9 else '⚠泄漏'}")

print("\n== FWAttention 自检 ==")
print("检查第一个块 (st=0, memories=None) 是否正确跳过 fast-weight 检索 (无条件泄漏):")
