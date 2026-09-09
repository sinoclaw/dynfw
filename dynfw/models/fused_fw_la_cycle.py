"""DynFW 蒸馏版 v5：循环潜推理 + BDH 完整读回（对齐 bdh_qwen 94.97）。

前提（父亲批准 + 军规归因）：la_cycle(~129) 距 bdh_qwen(94.97) 的差距【不是】初始化/正则，
而是【读回结构】——BDHQwen 用：
  x_sparse = ReLU(x@encoder)
  yKV = attn(x_sparse, x_sparse, V=x)        # K-is-Q, V=原始隐藏
  y_sparse = ReLU(yKV @ encoder_v)            # 第二稀疏投影
  xy = x_sparse * y_sparse                    # 双稀疏通道门控 (BDH 承重件)
  out = xy @ decoder                          # decoder 读出
  x = ln(x + out)
而 v4(la_cycle) 只有 attn + out Linear，缺第二投影/双门控/decoder。

v5 = 把 BDH 完整 block 当作【循环潜推理的一步】：每步 latent 喂回 -> 完整 BDH block -> 迭代。
单变量：与 bdh_qwen 同 block 结构；唯一新增 = 循环潜推理(steps) 潜状态喂回。
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class Attention(nn.Module):
    """BDH K-is-Q 因果线性注意（RoPE 位置 + 保顺序），同 bdh_qwen。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = Attention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
                    .view(1, 1, -1, 1)) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockCycle(nn.Module):
    """单个 BDH block（对齐 bdh_qwen）+ 支持循环潜推理接口。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=4):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(cfg)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(0.0)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """x: [B,1,T,D] (BDH 单隐状态，可作循环潜推理的 latent)。返回同形状。"""
        C = self.config
        B = x.shape[0]; T = x.shape[2]
        D = self.D; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.ln(x)
        x_latent = x @ self.encoder            # [B,nh,T,N]
        x_sparse = F.relu(x_latent)
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)   # V=原始x
        yKV = self.ln(yKV)
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = self.drop(xy_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        x = self.ln(x + y)
        return x

    def np(self):
        return sum(p.numel() for p in self.parameters())


class BDHBlockCycleLM(nn.Module):
    """LM 封装：embed -> 循环潜推理(BDH block 迭代) -> head。Qwen3 vocab 兼容。"""
    def __init__(self, D=128, N=None, k=16, nh=4, vocab=151936, use_ffn=True, n_layer=1, steps=4, mlp_mult=128):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.head = nn.Linear(D, vocab, bias=False)
        # 每层一个 BDH-block; 循环潜推理 = 同一 block 重复 steps 次(潜状态喂回)
        self.blocks = nn.ModuleList([BDHBlockCycle(D, nh, mlp_mult, vocab, steps) for _ in range(n_layer)])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, t=None):
        B, T = x.size()
        h = self.e(x).unsqueeze(1)           # [B,1,T,D]
        for block in self.blocks:
            for _ in range(self.steps):
                h = block(h)                 # 循环潜推理: 潜状态喂回迭代
        logits = self.head(h.view(B, T, self.D))
        loss = None if t is None else F.cross_entropy(logits.view(-1, self.vocab), t.view(-1))
        return logits, loss

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
