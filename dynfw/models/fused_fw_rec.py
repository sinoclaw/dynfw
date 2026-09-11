"""DynFW 蒸馏版 v2：递推快权 rho 记忆语言模型（Qwen3 vocab 兼容）。

方案A 方向A 改进（2026-09-08）。机制改动：rho 从"整段求和"（无时序袋状）
改成"逐 token 递推累积"（RNN/线性注意力式快权），使每个位置能读到
【截止当前】的历史，从而保留顺序信息。这是 fast-weights 的本意。

对比原 FusedFWQwen 的唯一区别：rho 更新方式（求和 -> 递推 cumsum），
其余（嵌入/D/N/k/FFN/head）完全一致，保证变量隔离。

forward 核心（递推 rho，向量化 cumsum）：
  act_t = topk(enc(et)) 稀疏激活 [B,T,N]
  ln_et_t = LN(et)                              [B,T,D]
  rho_cum[t] = Σ_{s<=t} act[s] ⊗ ln_et[s]      [B,T,N,D] 前缀和(含当前，保留顺序)
  mem_ctx[t] = act[t] · rho_cum[t]              [B,T,D]
  h = et + mem_ctx  ->  LN  ->  out  ->  (+FFN)
"""
import torch, torch.nn as nn, torch.nn.functional as F


class FusedFWRecurrent(nn.Module):
    """递推快权 rho 记忆的语言模型（Qwen3 vocab 兼容）。"""
    def __init__(self, D=128, N=512, k=16, vocab=151936, use_ffn=True, n_layer=1):
        super().__init__()
        self.D = D; self.N = N; self.k = k; self.vocab = vocab
        self.use_ffn = use_ffn; self.n_layer = n_layer
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, vocab, bias=False)
        self.encs = nn.ModuleList([nn.Linear(D, N, bias=False) for _ in range(n_layer)])
        self.decs = nn.ModuleList([nn.Linear(N, D, bias=False) for _ in range(n_layer)])
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    def forward(self, x, t=None):
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h, _, _ = recurrent_rho(self, h, self.encs[i], self.decs[i], self.out)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_hidden(self, x):
        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h, _, _ = recurrent_rho(self, h, self.encs[i], self.decs[i], self.out)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        return self.ln(h)

    def head_params(self):
        """返回 (weight(V,D), bias(V,) 或 None)，与 forward 中 head 投影严格一致。"""
        return self.head.weight, None
    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())


def recurrent_rho(self, et, enc, dec, out, norm_denom=True, eps=1e-6):
    """单层：逐 token 递推累积 rho（保留顺序），读数用累积平均防数值爆炸。

    v2（方向A+）：给 rho 加归一化分母（linear attention 标准做法）。
    rho_cum[t] = Σ_{s<=t} act[s]⊗ln_et[s]      [B,T,N,D] 前綴和(含当前)
    denom[t]   = Σ_{s<=t} act[s]               [B,T,N]   每槽累积激活量
    mem_ctx[t] = act[t] · (rho_cum[t]/denom[t])  # 累积平均，数值稳定，保留顺序
    """
    B, T, D = et.size(); N = self.N; k = self.k
    lat = enc(et)                                    # [B,T,N]
    topv, topi = torch.topk(lat, k, dim=-1)
    act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # [B,T,N] 稀疏
    ln_et = self.ln(et)                              # [B,T,D]
    # 递推累积 rho：rho_cum[t] = Σ_{s<=t} act[s] ⊗ ln_et[s]（含当前，保留先后顺序）
    prod = act.unsqueeze(-1) * ln_et.unsqueeze(2)    # [B,T,N,D]
    rho_cum = torch.cumsum(prod, dim=1)              # [B,T,N,D] 前綴和
    if norm_denom:
        # 归一化分母：每槽累积激活量，把"纯累加"变"累积平均"，防数值爆炸
        denom = torch.cumsum(act, dim=1)             # [B,T,N]
        denom = denom.clamp_min(eps)                 # 防除零
        rho_norm = rho_cum / denom.unsqueeze(-1)     # [B,T,N,D]
    else:
        rho_norm = rho_cum
    # 读回：mem_ctx[t] = act[t] · rho_norm[t]
    mem_ctx = torch.einsum('btn,btnd->btd', act, rho_norm)  # [B,T,D]
    h = et + mem_ctx; h = self.ln(h); h = out(h)
    return h, act, rho_norm
