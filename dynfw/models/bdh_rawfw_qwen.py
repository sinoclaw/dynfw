"""DynFW 蒸馏版 BDHRawFWQwen：基于 BDHQwen(bdh_qwen.py, raw整段 O(T²)) 的 raw 分块 + fast-weight 版。
唯一差异 = Attention 换成 BDH-CQ 官方 raw 分块(无softmax, tril-1, mask=0) + 跨块 fast-weight(O(T))。
其余(weight-shared循环/encoder/decoder/encoder_v/双向门控/残差LN/embed/lm_head)与 BDHQwen 完全一致。

单变量对照: BDHRawFWQwen(W=block=256, 每序列1块) ≈ BDHQwen(88.41); 触发跨块(W<block) 则 O(T)。
"""
import dataclasses, math
import torch, torch.nn as nn, torch.nn.functional as F


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 151936


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class FWAttentionRaw(nn.Module):
    """BDH-CQ 官方 raw 分块注意(无 softmax) + 跨块 fast-weight 记忆累积(O(T), 无 T² 矩阵)。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = FWAttentionRaw.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N] (K is Q); V:[B,1,T,D]; memories: 可选[B,nh,N,D]历史fast-weight。
        返回 (out[B,nh,T,D], new_mem[B,nh,N,D])。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N = self.N; D = self.D
        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)
        KR = QR
        out_chunks = []
        new_mem = memories if memories is not None else torch.zeros(B, nh, N, D, device=Q.device)
        for st in range(0, T, W):
            en = min(st + W, T)
            q_c = QR[:, :, st:en]
            k_c = KR[:, :, st:en]
            v_c = V[:, :, st:en]
            w = en - st
            sim = q_c @ k_c.mT              # [B,nh,w,w]
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=-1)
            attn = sim.masked_fill(~causal, 0.0)   # raw 无 softmax -> 未来=真0, 严格因果
            agg = attn @ v_c                # [B,nh,w,D]
            if new_mem is not None and (st > 0 or memories is not None):
                retr = torch.einsum('bhwd,bhde->bhwe', q_c, new_mem)
                agg = agg + retr
            out_chunks.append(agg)
            new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c, v_c)  # O(T)累积
        out = torch.cat(out_chunks, dim=2)
        return out, new_mem


class BDHRawFWQwen(nn.Module):
    """BDHQwen 的 raw 分块 + fast-weight 版（其余结构不变）。"""
    def __init__(self, D=128, n_layer=2, nh=4, mlp_mult=128, vocab=151936,
                 dropout=0.0, W=512):
        super().__init__()
        cfg = BDHConfig(n_layer=n_layer, n_embd=D, dropout=dropout,
                        n_head=nh, mlp_internal_dim_multiplier=mlp_mult,
                        vocab_size=vocab)
        self.config = cfg
        self.D = D; self.vocab = vocab; self.W = W
        nh = cfg.n_head
        N = cfg.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = FWAttentionRaw(cfg)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(vocab, D)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, vocab)).normal_(std=0.02))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        for _ in range(C.n_layer):
            mem = None  # 每层独立, 层内块间 fast-weight (W=block时1块无跨块, mem不影响 → ≈BDHQwen)
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV, mem = self.attn(Q=x_sparse, K=x_sparse, V=x, memories=mem, W=self.W)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
