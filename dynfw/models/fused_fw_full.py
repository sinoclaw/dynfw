"""DynFW 蒸馏版 v4：FusedFW + BDH 完整机制融合学生模型（Qwen3 vocab 兼容）。

方案"吸收机制 · 完整版"（2026-09-08）。从 BDH 吸收全部关键取胜机制：
  1. 超大稀疏隐空间（ReLU 正激活，N = mult*D//nh）
  2. K-is-Q 因果注意（保顺序 + 内容寻址）
  3. encoder_v 二阶段（内容经注意读出后再投影）
  4. 连续通道乘法门控（x_sparse * y_sparse）—— BDH 的核心"动态性"
  5. decoder 读回 + LayerNorm 残差

对比系列（同 FusedFW 骨架，只升级读回/门控机制）：
  - FusedFW        : einsum 求和 rho（丢顺序）            -> 362.67
  - FusedFW_rec    : cumsum 累加（保顺序/无差别）          -> 307.22
  - FusedFW_la     : 内容寻址因果读回（单头简化）          -> 254.31
  - FusedFWFull    : BDH 完整机制（通道门控 + encoder_v） -> 目标逼近 100
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from dynfw.models.bdh_qwen import get_freqs  # 复用频率


class CausalAttention(nn.Module):
    """BDH 式 K-is-Q 内容寻址因果注意。

    use_softmax=False（默认）：raw attention，直接 scores @ V（BDH 原版，无归一化，
        靠 Pre-LN + Q=K 自相似矩阵的天然性质，见 vision-bdh tazken 的消融发现）。
    use_softmax=True：加 softmax（标准 Transformer 口径），用于消融验证
        「softmax 是否与 Pre-LN 冲突」—— 判据：若加了反而变差，则证 BDH 社区的
        Pre-LN+raw-attention 协同在语言建模蒸馏上也成立。
    """
    def __init__(self, nh, N, use_softmax=False):
        super().__init__()
        self.nh = nh
        self.use_softmax = use_softmax
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = CausalAttention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, V):
        """Q: [B,nh,T,N] 稀疏激活(作Q&K), V: [B,1,T,D]。返回 [B,nh,T,D]。"""
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        scores = (QR @ QR.mT).tril(diagonal=-1)
        if self.use_softmax:
            scores = F.softmax(scores, dim=-1)
        return scores @ V


class FusedFWFull(nn.Module):
    """FusedFW 骨架 + BDH 完整机制（通道门控 + encoder_v + 大稀疏维）。

    tie=False（默认）：embed 与 lm_head 独立两份。
    tie=True：lm_head 复用 embed.weight（tie embeddings），省 vocab×D 参数。
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
        self.head = nn.Linear(D, vocab, bias=False)
        if tie:
            # 让 lm_head 与 embed 共享同一权重（并冻结独立 head 的前向）
            self.head.weight = self.e.weight
        # BDH 完整机制：每层一套 encoder/decoder/encoder_v/attn
        self.encoders = nn.ParameterList([
            nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)) for _ in range(n_layer)])
        self.decoders = nn.ParameterList([
            nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02)) for _ in range(n_layer)])
        self.encoder_vs = nn.ParameterList([
            nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02)) for _ in range(n_layer)])
        self.attns = nn.ModuleList([CausalAttention(nh, N, use_softmax=self.use_softmax) for _ in range(n_layer)])
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    def forward(self, x, t=None):
        B, T = x.size(); D = self.D; nh = self.nh
        h = self.e(x).unsqueeze(1)          # [B,1,T,D]
        h = self.ln(h)
        for i in range(self.n_layer):
            h = bdh_full_layer(self, h, self.encoders[i], self.decoders[i],
                               self.encoder_vs[i], self.attns[i])
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        lg = h.view(B, T, D) @ self.head.weight.t()
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def forward_hidden(self, x):
        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)（分块 KL 入口）。"""
        B, T = x.size(); D = self.D; nh = self.nh
        h = self.e(x).unsqueeze(1)          # [B,1,T,D]
        h = self.ln(h)
        for i in range(self.n_layer):
            h = bdh_full_layer(self, h, self.encoders[i], self.decoders[i],
                               self.encoder_vs[i], self.attns[i])
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        return h.view(B, T, D)

    def head_params(self):
        """(weight(V,D), None)，与 forward 中 @ self.head.weight.t() 严格一致。"""
        return self.head.weight, None

    def np(self):
        return sum(p.numel() for p in self.parameters())


def bdh_full_layer(self, x, encoder, decoder, encoder_v, attn):
    """BDH 完整单层：稀疏激活 -> K-is-Q 因果注意 -> encoder_v 二阶段 -> 通道门控 -> decoder。"""
    B, _, T, D = x.size(); nh = self.nh; N = self.N
    # 1) 稀疏激活：x @ encoder -> [B,nh,T,N]（torch 广播），ReLU 正激活
    x_latent = x @ encoder                                # [B,1,T,D] @ [nh,D,N] -> [B,nh,T,N]
    x_sparse = F.relu(x_latent)                            # 超大稀疏
    # 2) K-is-Q 因果注意：sparse 作 Q/K，x 作 V
    yKV = attn(x_sparse, x)                                # [B,nh,T,D] 注意内容寻址读历史
    yKV = self.ln(yKV)
    # 3) encoder_v 二阶段：yKV 再投影到稀疏空间
    y_latent = yKV @ encoder_v                             # [B,nh,T,D] @ [nh,D,N] -> [B,nh,T,N]
    y_sparse = F.relu(y_latent)
    # 4) 连续通道乘法门控（BDH 核心动态性）
    xy_sparse = x_sparse * y_sparse                        # [B,nh,T,N]
    # 5) decoder 读回：xy_sparse [B,nh,T,N] -> 组合到 [B,T,N*nh] @ decoder[N*nh, D]
    xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)  # [B,1,T,N*nh]
    yMLP = xy_flat @ decoder                               # [B,1,T,D]
    y = self.ln(yMLP)
    x = self.ln(x + y)
    return x
