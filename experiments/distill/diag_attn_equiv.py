# 诊断: v6 FWAttention vs DLA DLAFastAttn 前向是否等价
# 同随机输入, 同配置, 对比逐块输出 diff -> 定位 DLA 比 v6 差的根因
import torch, sys, math
sys.path.insert(0, '/data/dynfw')

from dynfw.models.fused_fw_fw_cycle import FWAttention, Config as C6
from dynfw.models.fused_fw_dla_cycle import DLAFastAttn, Config as CD

torch.manual_seed(0)
D, nh, N = 128, 16, 512   # mlp_mult=64 -> N = 64*128//16 = 512
W = 256
T = 256

def make_cfg(Cls):
    return Cls(1, D, nh, 64, 151936)

fa = FWAttention(make_cfg(C6))
dla = DLAFastAttn(make_cfg(CD))
dla.cache.K = 8

# 同随机输入
torch.manual_seed(1)
x = torch.randn(2, 1, T, D)
x_sparse = torch.randn(2, nh, T, N)

# 单层 (memories=None) 前向
out_v6, mem_v6 = fa(Q=x_sparse, K=x_sparse, V=x, memories=None, W=W)
out_dla, mem_dla = dla(Q=x_sparse, K=x_sparse, V=x, memories=None, W=W)
print(f"[单层] v6 out {tuple(out_v6.shape)}  dla out {tuple(out_dla.shape)}")
diff = (out_v6 - out_dla).abs()
print(f"[单层] maxdiff={diff.max().item():.8f}  meandiff={diff.mean().item():.8f}")
print("[单层] meld 逐块 maxdiff:")
for st in range(0, T, W):
    en = min(st + W, T)
    print(f"   block {st:4d}-{en:4d}: maxdiff={diff[:, :, st:en].max().item():.6e}")

# 检查 DLA 合并是否触发: 看 used
S, I, n, used = mem_dla
print(f"[单层] DLA used={used}, K={dla.cache.K}  -> 合并是否触发: {used >= dla.cache.K}")

# 检索贡献对比: v6 的 new_mem vs DLA 的 S[:used].sum
print(f"[单层] v6 new_mem shape {tuple(mem_v6.shape)}, DLA S[:used] sum shape "
      f"{tuple(S[:, :, :used].sum(dim=2).shape)}")
print(f"[单层] new_mem vs S.sum 差: {(mem_v6 - S[:, :, :used].sum(dim=2)).abs().max().item():.8f}")
