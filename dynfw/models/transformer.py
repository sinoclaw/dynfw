"""DynFW 参考基线：SDPA-Transformer（现代最佳实现基线）。

v0.0.1 冻结实现（源自 fdn 仓库 fair_bench_v2.py，commit 97f3eaf 线）。

用 PyTorch SDPA（scaled_dot_product_attention, is_causal）+ 正弦固定位置编码（0 参数字典）。
对比时必须用此基线（而非 naive attention），否则会夸大 FusedFW 的相对优势（见 docs/fairness.md）。
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

MAXT = 8192


class TF_sdpa(nn.Module):
    def __init__(self, D=128, nh=4, n_layer=2, vocab=256, maxT=MAXT):
        super().__init__(); self.vocab = vocab; self.D = D; self.nh = nh; self.n_layer = n_layer
        assert D % nh == 0
        self.e = nn.Embedding(vocab, D)
        # 正弦固定位置编码（0 参数字典，避免 maxT*D 参数池污染'同参数'对齐）
        pos = torch.zeros(maxT, D); ar = torch.arange(maxT).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-np.log(10000.0) / D))
        pos[:, 0::2] = torch.sin(ar * div); pos[:, 1::2] = torch.cos(ar * div)
        self.register_buffer('pos', pos)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleList([nn.LayerNorm(D), nn.Linear(D, 3*D), nn.Linear(D, D),
                nn.LayerNorm(D), nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D)]))
        self.ln = nn.LayerNorm(D); self.h = nn.Linear(D, vocab)

    def forward(self, x, t=None):
        B, T = x.size(); h = self.nh; D = self.D
        y = self.e(x) + self.pos[:T].unsqueeze(0)
        for ln1, qkv, proj, ln2, w1, act, w2 in self.blocks:
            xx = ln1(y); q, k, v = qkv(xx).chunk(3, dim=-1)
            q = q.view(B, T, h, D//h).transpose(1, 2); k = k.view(B, T, h, D//h).transpose(1, 2)
            v = v.view(B, T, h, D//h).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # SDPA
            o = o.transpose(1, 2).contiguous().view(B, T, D); o = proj(o)
            y = y + o; y = y + w2(act(w1(ln2(y))))
        lg = self.h(self.ln(y))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_hidden(self, x):
        """蒸馏用：返回 lm_head 之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""
        B, T = x.size(); h = self.nh; D = self.D
        y = self.e(x) + self.pos[:T].unsqueeze(0)
        for ln1, qkv, proj, ln2, w1, act, w2 in self.blocks:
            xx = ln1(y); q, k, v = qkv(xx).chunk(3, dim=-1)
            q = q.view(B, T, h, D//h).transpose(1, 2); k = k.view(B, T, h, D//h).transpose(1, 2)
            v = v.view(B, T, h, D//h).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).contiguous().view(B, T, D); o = proj(o)
            y = y + o; y = y + w2(act(w1(ln2(y))))
        return self.ln(y)

    def head_params(self):
        """(weight(V,D), bias(V,))，与 forward 中 self.h(self.ln(y)) 严格一致。"""
        return self.h.weight, self.h.bias

    def np(self):
        return sum(p.numel() for p in self.parameters())


def tf_decode_kvcache(model, prompt, gen=16):
    """TF 带 KV-cache 逐 token 解码（attention 只对已缓存 k,v 计算，decode 是 O(T)/token）。"""
    model.eval(); D = model.D; nh = model.nh; hd = D // nh
    with torch.no_grad():
        caches = []; tok = prompt
        for _ in range(gen):
            T = tok.shape[1]
            y = model.e(tok) + model.pos[:T].unsqueeze(0)
            for bi, (ln1, qkv, proj, ln2, w1, act, w2) in enumerate(model.blocks):
                xx = ln1(y); q, k, v = qkv(xx).chunk(3, dim=-1)
                q = q.view(1, T, nh, hd).transpose(1, 2); k = k.view(1, T, nh, hd).transpose(1, 2)
                v = v.view(1, T, nh, hd).transpose(1, 2)
                if bi >= len(caches):
                    caches.append((k, v))
                else:
                    caches[bi] = (torch.cat([caches[bi][0], k], dim=2), torch.cat([caches[bi][1], v], dim=2))
                ck, cv = caches[bi]
                o = F.scaled_dot_product_attention(q, ck, cv, is_causal=True)
                o = o.transpose(1, 2).contiguous().view(1, T, D); o = proj(o)
                y = y + o; y = y + w2(act(w1(ln2(y))))
            tok = model.h(model.ln(y))[:, -1:].argmax(-1).to(torch.long)
    return tok
