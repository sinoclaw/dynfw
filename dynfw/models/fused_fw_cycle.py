"""DynFW 蒸馏版：循环潜推理 FusedFW v2（真正还原 BDH-CQ 的跨步记忆累积）。

v1 教训：我把循环潜推理接成了"重复前向"（每步 rho 重算），丢了 BDH-CQ 精髓
  —— memories 在步间【跨步累积】。v2 修正：rho 在步间延续累积（Hebbian），
  每步用读回后的潜状态 h 再喂回 + 更新 rho。

BDH-CQ 蓝本（BDHReasoningWrapper）：
  让步: latent=memories.embeds 喂回 -> 更新 memories(fast-weight 跨步累积)
        -> latent=memories.embeds -> 重复 steps 次
本变体等价实现：
  让步: h=读回潜状态喂回 -> rho = rho_old + act⊗ln_et（跨步累积）
        -> h'=读回(act, rho) -> 重复 steps 次
单变量隔离：除"循环潜推理+跨步rho"外，其它与 FusedFWQwen 完全相同。
"""
import torch, torch.nn as nn, torch.nn.functional as F


class FusedFWCYBLE(nn.Module):
    """循环潜推理 FusedFW v2（Qwen3 vocab 兼容，蒸馏接口）。"""
    def __init__(self, D=128, N=512, k=16, vocab=151936, use_ffn=True, n_layer=1, steps=4):
        super().__init__()
        self.D = D; self.N = N; self.k = k; self.vocab = vocab
        self.use_ffn = use_ffn; self.n_layer = n_layer; self.steps = steps
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
            h = self.cyclic_sparse_rho(h, self.encs[i], self.decs[i], self.out, self.steps)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def cyclic_sparse_rho(self, et, enc, dec, out, steps):
        """循环潜推理 v2: rho 跨步累积(B=1, T固定, rho [B,N,D] 持久跨步)。"""
        B, T, D = et.size(); N = self.N; k = self.k
        # 第一步: 初始化 rho（从当前注入）
        h = et
        rho = torch.zeros(B, N, D, device=et.device)   # 跨步累积的记忆
        for _ in range(steps):
            lat = enc(h)
            topv, topi = torch.topk(lat, k, dim=-1)
            act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # [B,T,N]
            ln_h = self.ln(h)
            # 跨步累积: rho = rho_old + Σ_t act_t ⊗ ln_et_t
            rho = rho + torch.einsum('btn,btd->bnd', act, ln_h)
            # 读回: mem_ctx[t] = act[t] · rho（用累积后的 rho）
            mem_ctx = torch.einsum('btn,bnd->btd', act, rho)
            h_next = h + mem_ctx
            h_next = self.ln(h_next)
            h = out(h_next)
        return h

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
