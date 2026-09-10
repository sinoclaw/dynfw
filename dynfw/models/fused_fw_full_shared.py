"""DynFW 蒸馏版 v5：FusedFWFull + BDH 省参机制（weight-sharing + tie embeddings）。

v5 = 在 v4(FusedFWFull) 基础上，吸收官方 BDH 两个关键省参机制，解决「层太宽堆不深 + 词表税虚胖」：
  1. **weight-sharing**：单套 encoder/decoder/encoder_v/attn，n_layer 次循环复用
     （官方 BDH n_layer=8 用 1 套参数循环 8 次，省了 8 倍的大头）。
  2. **tie embeddings**：embed 与 lm_head 共享同一权重（word_embeddings tied），
     消灭 vocab×D 的重复词表税（官方 BDH 用 0.2M 管住词表，我们 v4 花了 233M）。

保留 v4 已验证的承重配置：
  - raw attention（no softmax）+ Pre-LN（vision-bdh 协同，softmax 消融证承重）
  - K-is-Q 内容寻址因果注意 + 通道门控 + encoder_v 二阶段 + LN 残差

对比：
  - FusedFWFull(v4): 每层独立参数 -> 4层 385M（词表税 233M，未tie）
  - FusedFWFullShared(v5): 单套循环 + tied -> 同样配比参数大幅下降，深度可真正堆上去
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from dynfw.models.fused_fw_full import CausalAttention  # 复用 raw/softmax 注意 + RoPE


class FusedFWFullShared(nn.Module):
    """FusedFW 骨架 + BDH 完整机制 + weight-sharing（单套循环），默认不tie词表。

    控制变量（判据先行）：本版只做 weight-sharing、不 tie —— 对齐 BDH 成功配方
    （weight-sharing ✓ + 不tie ✓，见上一轮 v5 崩因= tie 拖累的消融结论）。
    参数不随 n_layer 增长（层数只进循环不进参数），同配比 28 层参数恒定，深度白送。
    """
    def __init__(self, D=128, N=512, k=16, nh=4, mlp_mult=128,
                 vocab=151936, use_ffn=False, n_layer=1, use_softmax=False, tie=False):
        super().__init__()
        self.D = D; self.N = N; self.k = k; self.nh = nh; self.vocab = vocab
        self.use_ffn = use_ffn; self.n_layer = n_layer; self.use_softmax = use_softmax
        self.tie = tie
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        # 关键：单套参数（weight-sharing），n_layer 只控制循环次数，不增加参数
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = CausalAttention(nh, N, use_softmax=self.use_softmax)
        if tie:
            self.lm_head = None           # 用 embed.weight 转置，不额外建
        else:
            self.lm_head = nn.Parameter(torch.zeros((D, vocab)).normal_(std=0.02))
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    def forward(self, x, t=None):
        B, T = x.size(); D = self.D
        h = self.e(x).unsqueeze(1)          # [B,1,T,D]
        h = self.ln(h)
        for _ in range(self.n_layer):        # 循环复用同一套参数（weight-sharing）
            h = bdh_full_layer(self, h, self.encoder, self.decoder, self.encoder_v, self.attn)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        # lm_head：tie 则用 embed.weight 转置，否则用独立参数
        if self.tie:
            lg = h.view(B, T, D) @ self.e.weight.t()
        else:
            lg = h.view(B, T, D) @ self.lm_head
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())


def bdh_full_layer(self, x, encoder, decoder, encoder_v, attn):
    """BDH 完整单层（同 v4 逻辑）：稀疏激活 -> K-is-Q 因果注意 -> encoder_v -> 通道门控 -> decoder。"""
    B, _, T, D = x.size(); nh = self.nh; N = self.N
    x_latent = x @ encoder                                # [B,1,T,D] @ [nh,D,N] -> [B,nh,T,N]
    x_sparse = F.relu(x_latent)                            # 超大稀疏
    yKV = attn(x_sparse, x)                                # [B,nh,T,D] 注意内容寻址读历史
    yKV = self.ln(yKV)
    y_latent = yKV @ encoder_v                             # [B,nh,T,D] @ [nh,D,N] -> [B,nh,T,N]
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse                        # 连续通道乘法门控（BDH 核心动态性）
    xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)  # [B,1,T,N*nh]
    yMLP = xy_flat @ decoder                               # [B,1,T,D]
    y = self.ln(yMLP)
    x = self.ln(x + y)
    return x
