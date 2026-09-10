"""DynFW 蒸馏版 v5：线性注意力 + 内容寻址读回（FusedFW_lin）。

方案"线性和保顺序"（2026-09-08）。目标：同时满足 能力不输 + 推理降本。

解决核心矛盾：
  - FusedFW 的 rho 是 O(1) 固定状态（线性、decode常数内存）但"丢顺序"（整段求和）
  - BDH 的"内容寻址因果注意"保序但 O(T²)（QR@QR.T），FusedFWFull 继承 O(T²) 变慢

本模型 = 线性注意力（Katharopoulos-style）+ FusedFW 骨架：
  S_t = S_{t-1} + k_t^T·v_t        # O(1) 状态，cumsum 实现，线性
  out_t = q_t · S_t                # 每位置 query 内容寻址题问状态（保顺序）
  
证据（预期）：
  - prefill 扫 T wall-clock 翻倍比 ~1.0 → 真 O(T) 线性（比 BDH 的 ~2.0 = O(T²) 硬性更优）
  - decode 每 token 常数内存、不随 T 增长

结构：嵌入 → 投影 Q/K/V → 线性注意状态累积 → 通道读回 → FFN → head
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


class FusedFWLin(nn.Module):
    """线性注意力 + FusedFW 骨架（Qwen3 vocab 兼容）。"""
    def __init__(self, D=128, nh=4, dk=32, vocab=151936, n_layer=2, use_ffn=True):
        super().__init__()
        self.D = D; self.nh = nh; self.dk = dk; self.vocab = vocab
        self.n_layer = n_layer; self.use_ffn = use_ffn
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        # 每层：Q/K/V 投影 + 读回投影 + FFN
        self.qkvs = nn.ModuleList([
            nn.Linear(D, 2*nh*dk + D, bias=False) for _ in range(n_layer)])
        self.outer = nn.ModuleList([nn.Linear(nh*D, D, bias=False) for _ in range(n_layer)])
        if use_ffn:
            self.ffns = nn.ModuleList([
                nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
                for _ in range(n_layer)])

    def forward(self, x, t=None):
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h = self.lin_layer(h, self.qkvs[i], self.outer[i],
                               self.ffns[i] if self.use_ffn else None)
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def lin_layer(self, h, qkv, outer, ffn):
        """线性注意力层：S += kᵀv (cumsum)，out = q·S。"""
        B, T, D = h.size(); nh = self.nh; dk = self.dk
        qkv_out = qkv(h)                              # [B,T, 2nh*dk+D]
        q = qkv_out[..., :nh*dk].view(B, T, nh, dk).transpose(1, 2)   # [B,nh,T,dk]
        k = qkv_out[..., nh*dk:2*nh*dk].view(B, T, nh, dk).transpose(1, 2)
        v = qkv_out[..., 2*nh*dk:].view(B, T, D)
        # 状态累积 S_t = Σ_{s<=t} k_s ⊗ v_s  (linear attention，cumsum O(T))
        # k:[B,nh,T,dk] v:[B,T,D] -> 广播 v 到 nh 头
        v_b = v.unsqueeze(1).expand(-1, nh, -1, -1)      # [B,nh,T,D]
        kv = torch.einsum('bntk,bntd->bntkd', k, v_b)    # [B,nh,T,dk,D] 逐token k⊗v
        S = torch.cumsum(kv, dim=2)                    # [B,nh,T,dk,D] 状态累积
        # 内容寻址读回：out_t = q_t · S_t
        out = torch.einsum('bntk,bntkd->bntd', q, S)      # [B,nh,T,D]
        out = out.transpose(1, 2).reshape(B, T, nh*D)    # [B,T,nh*D]
        out = outer(out)                                 # [B,T,D]
        h = h + out
        h = F.layer_norm(h, (D,))
        if ffn is not None:
            h = h + ffn(h)
        return h

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
