"""DynFW 蒸馏版 BDH：Block-Diagonal Highway 学生模型（Qwen3 vocab 兼容）。

来源：官方 pathwaycom/bdh（The Dragon Hatchling, arXiv:2509.26507）的 BDH-GPU 教学实现。
封装改动（对接蒸馏管线）：
  1. vocab 参数化（default 151936 = Qwen3 tokenizer vocab_size）
  2. 加 forward_logits() / np() 接口（与其他学生架构对齐）
  3. mlp_internal_dim_multiplier 可配置（默认 128 -> N=4096 超大稀疏；可调小省显存）
保持 BDH 核心机制不变：
  - ReLU 稀疏激活（x_sparse = ReLU(x @ encoder)）超大稀疏隐空间
  - K-is-Q 因果线性注意（self-attention，保留顺序）
  - fast-weight 通道乘法门控（x_sparse * y_sparse）
  - decoder 读回，LayerNorm 残差
作用：作为"有顺序记忆 + 稀疏"的架构候选，与 FusedFW（丢顺序）/ TF_sdpa 作三方对轰。
"""
import dataclasses, math
import torch, torch.nn as nn, torch.nn.functional as F


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128   # 稀疏维 N = mult*D//nh
    vocab_size: int = 151936


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDHQwen(nn.Module):
    """BDH-GPU 学生模型（Qwen3 vocab 兼容，蒸馏接口）。"""
    def __init__(self, D=128, n_layer=2, nh=4, mlp_mult=128, vocab=151936,
                 dropout=0.0):
        super().__init__()
        cfg = BDHConfig(n_layer=n_layer, n_embd=D, dropout=dropout,
                        n_head=nh, mlp_internal_dim_multiplier=mlp_mult,
                        vocab_size=vocab)
        self.config = cfg
        self.D = D; self.vocab = vocab
        nh = cfg.n_head
        N = cfg.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(cfg)
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
        x = self.embed(idx).unsqueeze(1)      # B,1,T,D
        x = self.ln(x)
        for _ in range(C.n_layer):
            x_latent = x @ self.encoder        # B,nh,T,N
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
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

    def forward_hidden(self, idx):
        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""
        C = self.config
        B, T = idx.size()
        D = C.n_embd; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.embed(idx).unsqueeze(1)      # B,1,T,D
        x = self.ln(x)
        for _ in range(C.n_layer):
            x_latent = x @ self.encoder        # B,nh,T,N
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        return x.view(B, T, D)

    def head_params(self):
        """返回 (weight(V,D), bias(V,) 或 None)，与 forward 中 head 投影严格一致。"""
        return self.lm_head.t(), None          # Parameter(D,V) -> (V,D) 视图，梯度回传统一 Parameter
    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
