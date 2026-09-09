"""DynFW 蒸馏版：Fused Fast-Weight（稀疏 rho 记忆）语言模型 -- Qwen3 vocab 兼容。

方案A 蒸馏学生模型（2026-09-08）。
与 v0.0.1 冻结基线的区别：
  1. vocab 参数化（默认 151936 = Qwen3 tokenizer vocab_size），可直接对接 Qwen3 tokenizer
  2. 保留 rho 快权记忆（本体机制不动），可扩展 D/N/k 以对齐千问0.6B 容量

用途：作为千问0.6B 蒸馏的学生模型，用教师预计算 logits 训练（KL 蒸馏）。
结构（同 FusedFW）：
  嵌入 → enc(D→N) → top-k 稀疏激活 → rho = Σ_t act⊗ln_et → act 加权读回
      → +FFN(非线性读回) → head(→vocab)
"""
import torch, torch.nn as nn, torch.nn.functional as F


class FusedFWQwen(nn.Module):
    """稀疏激活 + fast-weight rho 记忆的语言模型（Qwen3 vocab 兼容）。"""
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
            h, _, _ = sparse_rho_forward(self, h, self.encs[i], self.decs[i], self.out)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_logits(self, x):
        """只返回 logits（蒸馏用，教师/学生都走这个接口）。"""
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())


def sparse_rho_forward(self, et, enc, dec, out):
    """单层：稀疏激活 + rho 记忆读回。返回 h, act, rho。"""
    B, T, D = et.size(); N = self.N; k = self.k
    lat = enc(et)
    topv, topi = torch.topk(lat, k, dim=-1)
    act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))
    ln_et = self.ln(et)
    rho = torch.einsum('btn,btd->bnd', act, ln_et)
    mem_ctx = torch.einsum('btn,bnd->btd', act, rho)
    h = et + mem_ctx; h = self.ln(h); h = out(h)
    return h, act, rho
