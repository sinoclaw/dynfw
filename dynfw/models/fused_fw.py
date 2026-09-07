"""DynFW 核心模块：Fused Fast-Weight（稀疏 rho 记忆）语言模型。

v0.0.1 冻结实现（源自 fdn 仓库 diag_gap.py，commit 97f3eaf 线的公平审计版）。

结构（Fused Fast-Weight computation + Persistent associative state）：
  嵌入 → enc(D→N) → top-k 稀疏激活 → rho = Σ_t act⊗ln_et (Hebbian 累积, 无时序袋状)
       → act 加权读回 → +FFN(非线性读回) → head(→vocab)

注：rho 是无时序选择记忆池，主场是序列/长上下文任务（见 docs/limitations.md）。
"""
import torch, torch.nn as nn, torch.nn.functional as F


def sparse_rho_forward(self, et, enc, dec, out):
    """单层：稀疏激活 + rho 记忆读回。返回 h, act, rho。"""
    B, T, D = et.size(); N = self.N; k = self.k
    lat = enc(et)                                    # B,T,N
    topv, topi = torch.topk(lat, k, dim=-1)
    act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # 稀疏正激活
    ln_et = self.ln(et)
    rho = torch.einsum('btn,btd->bnd', act, ln_et)   # Hebbian 累积（袋状，无时序）
    mem_ctx = torch.einsum('btn,bnd->btd', act, rho)  # act 加权读回
    h = et + mem_ctx; h = self.ln(h); h = out(h)
    return h, act, rho


class FusedFW(nn.Module):
    """稀疏激活 + fast-weight rho 记忆的语言模型（FP32，CPU 可跑）。"""
    def __init__(self, D=128, N=512, k=16, vocab=256, use_ffn=False, n_layer=1):
        super().__init__(); self.D = D; self.N = N; self.k = k; self.vocab = vocab
        self.use_ffn = use_ffn; self.n_layer = n_layer
        self.e = nn.Embedding(vocab, D); self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D, bias=False); self.head = nn.Linear(D, vocab)
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

    def np(self):
        return sum(p.numel() for p in self.parameters())


def fw_new_state(D, N):
    """增量 decode 的 running rho 状态（单样本 [N,D]）。"""
    return torch.zeros((N, D))


def fw_forward_single(self, x_tok, rho):
    """逐 token decode：从 running rho 状态增量读回。返回 logits[1,1,vocab], 新 rho。"""
    et = self.e(x_tok)                                # [1,1,D]
    lat = self.encs[0](et)                            # [1,1,N]
    topv, topi = torch.topk(lat, self.k, dim=-1)
    act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # [1,1,N]
    ln_et = self.ln(et)                               # [1,1,D]
    rho = rho + torch.outer(act[0].reshape(-1), ln_et[0].reshape(-1))  # [N,D]
    mem_ctx = act[0].reshape(1, -1) @ rho             # [1,N]@[N,D]=[1,D]
    h = self.ln(et[0] + mem_ctx); h = self.out(h)
    if self.use_ffn:
        h = h + self.ffn(self.ln(h))
    return self.head(self.ln(h)).unsqueeze(0), rho
