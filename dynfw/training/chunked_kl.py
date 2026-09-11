"""分块 KL 蒸馏损失（Chunked Linear KL：lm_head 投影 + KL 融合，不物化 B×T×V logits）

背景 / 为什么需要
----------------
大词表（V≈152K）logit 蒸馏一次性算全量 logits 时，同时要持有
  s_lg(B,T,V) fp32 + t_lg(B,T,V) fp32 + log_softmax/softmax/kl_div 中间量
→ 峰值 ~5×B×T×V×4 字节。实测 (V=151936, D=1024, B=8, T=1024)：峰值 33.1 GiB。
只做"沿 T 切片算 loss"而不切断 autograd，仍会保留每块 logits 的计算图，
峰值只降到 19.5 GiB；必须把 lm_head 投影一并融进分块函数、反向重算，
才能降到 7.3 GiB（chunk=64，实测）。

做法（与 Liger-Kernel fused_linear_jsd 同思路）
--------------------------------------------
  1) 学生主干只产出 hidden(B,T,D)（不物化 logits）
  2) 本模块的 autograd.Function 内按 T 分块做 lm_head 投影 + KL
  3) forward 不留 logits 的图；backward 按块重算，用手写梯度公式

数学口径（必须与全量实现完全一致）
--------------------------------
  loss = batchmean(KL(p_t || p_s)) * temp²
       = (1/B) * temp² * Σ_{b,t} Σ_v p_t(v) * (log p_t(v) - log p_s(v))
  ∂loss/∂z_s(b,t,v) = (temp / B) * (p_s(v) - p_t(v))
  —— KL 对 logits 的梯度只依赖两个概率分布本身，不含教师 logsumexp 项，
     因此可以逐块独立累加，分块与全量数值等价（实测偏差 ≤ 1e-7，fp32 舍入级）。

额外收益：teacher_logits 允许放在 CPU（块内搬到 GPU，用完即释放），
教师 logits 因此不占 GPU 常驻显存。
"""
import torch
import torch.nn.functional as F


class ChunkedLinearKL(torch.autograd.Function):
    """hidden @ weight.T (+bias) -> KL(p_t || p_s)，沿 T 分块，反向重算。"""

    @staticmethod
    def forward(ctx, hidden, weight, bias, t_lg, chunk, temp, n_batch):
        B, T, D = hidden.shape
        V = weight.shape[0]
        cdtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        total = hidden.new_zeros(()).to(cdtype)
        Wf = weight.to(cdtype)
        bf = bias.to(cdtype) if (bias is not None and bias.numel()) else None
        with torch.no_grad():
            for i in range(0, T, chunk):
                h = hidden[:, i:i + chunk].to(cdtype)
                z = h @ Wf.t()
                if bf is not None:
                    z = z + bf
                t = t_lg[:, i:i + chunk]
                if t.device != z.device:
                    t = t.to(z.device, non_blocking=True)
                t = t.to(cdtype)
                ls = F.log_softmax(z / temp, dim=-1)
                p = F.softmax(t / temp, dim=-1)
                total = total + F.kl_div(ls, p, reduction='sum') * (temp ** 2)
        ctx.save_for_backward(hidden, weight, bias, t_lg)
        ctx.chunk = int(chunk)
        ctx.temp = float(temp)
        ctx.n_batch = int(n_batch)
        del Wf, bf
        return total / n_batch

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, bias, t_lg = ctx.saved_tensors
        B, T, D = hidden.shape
        V, _ = weight.shape
        chunk, temp, n_batch = ctx.chunk, ctx.temp, ctx.n_batch
        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        cdtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32

        Wf = weight.to(cdtype)
        bf = bias.to(cdtype) if bias.numel() else None
        scale = float(grad_out) * temp / n_batch

        g_hidden = torch.zeros_like(hidden) if need_h else None
        g_W = torch.zeros((V, D), device=weight.device, dtype=cdtype) if need_w else None
        g_b = torch.zeros((V,), device=weight.device, dtype=cdtype) if (need_b and bf is not None) else None

        for i in range(0, T, chunk):
            h = hidden[:, i:i + chunk].to(cdtype)
            z = h @ Wf.t()
            if bf is not None:
                z = z + bf
            t = t_lg[:, i:i + chunk]
            if t.device != z.device:
                t = t.to(z.device, non_blocking=True)
            t = t.to(cdtype)
            p_s = F.softmax(z / temp, dim=-1)
            p_t = F.softmax(t / temp, dim=-1)
            g_z = (p_s - p_t) * scale                      # ∂loss/∂z_s，与全量口径一致
            if g_hidden is not None:
                g_hidden[:, i:i + chunk] = (g_z @ Wf).to(hidden.dtype)
            if g_W is not None:
                g_W += g_z.reshape(-1, V).t() @ h.reshape(-1, D)
            if g_b is not None:
                g_b += g_z.sum(dim=(0, 1))
            del p_s, p_t, g_z, z, h, t

        if g_W is not None:
            g_W = g_W.to(weight.dtype)
        if g_b is not None:
            g_b = g_b.to(bias.dtype)
        return g_hidden, g_W, g_b, None, None, None, None


def chunked_kl_loss(hidden, weight, bias, teacher_logits, chunk=256, temperature=1.0):
    """分块 KL 蒸馏损失（drop-in 替换全量 kl_loss）。

    hidden          : (B,T,D) 学生主干输出（lm_head 之前），需要 grad
    weight          : (V,D) lm_head 权重（nn.Linear.weight 或 Parameter 转置视图）
    bias            : (V,) 或 None
    teacher_logits  : (B,T,V) 教师 logits，任意精度/设备（可放 CPU）
    chunk           : 沿 T 的块大小；越小越省显存（实测 64 时峰值 7.3 GiB）
    """
    B = hidden.shape[0]
    if bias is None:
        bias = torch.zeros(0, device=hidden.device, dtype=torch.float32)
    return ChunkedLinearKL.apply(hidden, weight, bias, teacher_logits,
                                 int(chunk), float(temperature), int(B))


def full_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """全量 KL（= distill_qwen.py 原 kl_loss 口径），仅作分块版本的等价性对照。"""
    ls = F.log_softmax(student_logits / temperature, dim=-1)
    p = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(ls, p, reduction='batchmean') * (temperature ** 2)
