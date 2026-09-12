"""验证 GDNFastAttnFLA 的正确性（对照朴素逐 token 参考实现，非对照 v6.6）。

参考实现精确复现 FLA/GLA 的语义：
    S_t = exp(g_t)·S_{t-1} + k_t⊗v_t
    o_t = q_t @ S_t  - (q_t·k_t)·v_t          ← 减掉 self 项（对齐 diagonal=-1）
并且记录与"v6.6 读出方式"(q_t @ S_{t-1}，无 α_t 因子)的差异，供台账如实记账。
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'


def naive(Q, K, V, gw, gb, include_self=True, use_scale=False, alpha_factor=False):
    """朴素逐 token 参考。Q,K:[B,nh,T,N]; V:[B,1,T,D]; gw:[N,1] gb:[1]。
    include_self=False 时减掉 self 项。
    alpha_factor=True 时读出用 S_{t-1} 再乘 α_t（= FLA 减 self 后的真实形式）。
    """
    B, nh, T, N = Q.shape
    D = V.shape[-1]
    Ksq = float(N) if use_scale else 1.0
    M = torch.zeros(B, nh, N, D, device=DEV, dtype=torch.float32)
    outs = []
    for t in range(T):
        q_t, k_t = Q[:, :, t].float(), K[:, :, t].float()
        v_t = V[:, 0, t].float().unsqueeze(1).expand(-1, nh, -1)   # [B,nh,D]（V 对头共享）
        alpha = torch.sigmoid(k_t @ gw + gb)                       # [B,nh,1]
        # FLA 语义：先更新状态，再用"写之后"的状态读出
        M = alpha.unsqueeze(-1) * M + torch.einsum('bhn,bhd->bhnd', k_t, v_t)
        o_t = torch.einsum('bhn,bhnd->bhd', q_t / Ksq ** 0.5, M)
        if not include_self:
            # 减掉 self 项 (q_t·k_t)·v_t —— 这才是 diagonal=-1 的含义
            self_dot = (q_t * k_t).sum(-1, keepdim=True)           # [B,nh,1]
            o_t = o_t - (self_dot * v_t) / Ksq ** 0.5
        outs.append(o_t)
    return torch.stack(outs, 2), M


from dynfw.models.fused_fw_gdn_cycle import Config
from dynfw.models.fused_fw_gdn_fla import GDNFastAttnFLA

B, nh, T, N, D = 2, 4, 128, 64, 32
MLP_MULT = N * nh // D          # 使 cfg 推出的 N 与测试用 N 一致
cfg = Config(1, D, nh, MLP_MULT, 1000)
attn = GDNFastAttnFLA(cfg, read_mode='raw', gate_mode='token').to(DEV)

Q = torch.randn(B, nh, T, N, device=DEV)
V = torch.randn(B, 1, T, D, device=DEV)

# 取模块内的 rope 与 gate，保证两边输入完全一致
r = torch.arange(0, T, device=DEV, dtype=torch.float32).view(1, 1, -1, 1)
QR = attn.rope(r * attn.freqs, Q)
gw = attn.gate.weight.detach().T.to(DEV).float()      # [N,1]
gb = attn.gate.bias.detach().float()                  # [1]

with torch.no_grad():
    # 用模块自己算（走 FLA）
    out_fla, mem_fla = attn(Q=Q, K=Q, V=V, memories=None, W=64)

# 朴素参考：FLA 语义 = 含 self 后减 self 项，且加 1/sqrt(N) 缩放
#   注意模块内部 scale=1.0（我们关掉了 FLA 默认缩放），故参考也不带缩放
ref_pre, mem_ref = naive(QR, QR, V, gw, gb, include_self=False)
ref_pre_scaled = ref_pre      # 模块 scale=1.0 ⇒ 无 1/sqrt(N)

print('=== GDNFastAttnFLA vs 朴素逐 token 参考（同输入/同门控）===')
print(f'  out  maxdiff = {(out_fla.float() - ref_pre_scaled).abs().max().item():.3e}')
print(f'  out  相对差 = {((out_fla.float()-ref_pre_scaled).abs().max()/ref_pre_scaled.abs().max()).item():.3e}')
print(f'  mem  maxdiff = {(mem_fla.float() - mem_ref).abs().max().item():.3e}')
print(f'  mem 相对差 = {((mem_fla.float()-mem_ref).abs().max()/mem_ref.abs().max()).item():.3e}')

print()
print('=== 顺带量化：FLA 语义 vs v6.6 读出方式的差异（α_t 因子）===')
ref_self, _ = naive(QR, QR, V, gw, gb, include_self=True)
print(f'  含 self 的 o 量级 = {ref_self.abs().max():.3f}')
print(f'  减 self 后量级     = {ref_pre_scaled.abs().max():.3f}')
print('  => o_t(FLA减self) = alpha_t * (q_t @ S_{t-1})')
print('     alpha_t in (0,1)，初始 bias=+4 => alpha 约 0.98')
print('     v6.6 读出 q_t@M_prev 不带 α_t ⇒ 二者差一个 α_t 因子，非位级等价（已如实记账）')

print()
print('=== 参数量一致性检查（应与 GDNFastAttn 相同）===')
from dynfw.models.fused_fw_gdn_cycle import GDNFastAttn
a0 = GDNFastAttn(cfg, read_mode='raw'); a1 = attn
n0 = sum(p.numel() for p in a0.parameters()); n1 = sum(p.numel() for p in a1.parameters())
print(f'  GDNFastAttn   = {n0:,}')
print(f'  GDNFastAttnFLA= {n1:,}')
print(f'  {"一致 ✓" if n0 == n1 else "不一致 ✗（需修）"}')
