"""v6 融合实现版（方案 A）：与 fused_fw_fw_cycle.py **数学等价**，只换实现形态。

对应 profile 定位的根因，逐条优化：
  1. 块内注意改 `F.scaled_dot_product_attention(is_causal=True)` —— 1 个融合内核
     替代 手写 masked_fill + softmax + matmul（原：无任何融合调用；TF：3 次 FlashAttention）
  2. 去掉 5 处 `.float()` 强制转换（原 profile: aten::copy_ 2610 次 / 38.2ms）
  3. 输出预分配 + 切片赋值，替代 `torch.cat(out_chunks)`（每次 forward 一次全量拷贝）
  4. rope 用切片赋值替代 `stack + view`（纯访存零计算）
  5. 分块循环内不再重建 causal mask（SDPA 的 is_causal 内置，无需物化 w×w 掩码）

用法：
    from dynfw.models.fused_fw_fw_cycle_opt import to_opt
    m = BDHBlockFWCycleLM(...)      # 基线模型（参数/结构完全一致）
    m = to_opt(m, strict_bf16=True) # 原地把每个 block 的 attn 换成融合实现

⚠️ 参数完全不变（只做类替换），state_dict 可直接互载 → 数值对照与训练对照都成立。
⚠️ 严格因果保持：块内 is_causal；跨块检索只用【已累积】的 memory（与因果修复后基线一致）。
"""
import torch
import torch.nn.functional as F

from dynfw.models.fused_fw_fw_cycle import FWAttention


class FWAttentionOpt(FWAttention):
    """融合实现。参数与 FWAttention 完全一致（本身不加任何参数）。"""

    # True = 全程 bf16（去 .float()）；False = 保留基线那几处 fp32 转换（更保守）
    strict_bf16 = True

    @staticmethod
    def rope_fast(phases, v):
        """等价于 FWAttention.rope，但用切片赋值避免 stack+view 的中间拷贝。"""
        pc, ps = FWAttention.phases_cos_sin(phases)
        out = (v * pc).to(v.dtype)
        rot = torch.empty_like(v)
        rot[..., 0::2] = -v[..., 1::2]
        rot[..., 1::2] = v[..., ::2]
        return out + (rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        bf16 = self.strict_bf16

        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope_fast(r * self.freqs, Q)

        out = torch.empty(B, nh, T, D, device=Q.device,
                          dtype=(Q.dtype if bf16 else torch.float32))
        new_mem = memories if memories is not None else torch.zeros(
            B, nh, N, D, device=Q.device, dtype=(Q.dtype if bf16 else torch.float32))

        for st in range(0, T, W):
            en = min(st + W, T)
            w = en - st
            q_c = QR[:, :, st:en]
            k_c = QR[:, :, st:en]
            v_c = V[:, :, st:en]                      # [B,1,w,D]
            sl = slice(st, en)

            # ① 块内因果注意：单个融合内核（等价于原 tril(diag=0)+softmax+matmul）
            #    ⚠️ 必须显式 scale=1.0 —— SDPA 默认会除以 sqrt(E) 缩放，而本架构
            #    （承袭 BDH/fast-weight 系）的块内注意**不做 √d 缩放**，不显式关掉就数值不等价。
            a = F.scaled_dot_product_attention(
                q_c, k_c, v_c.expand(B, nh, w, D), is_causal=True, scale=1.0)

            # ② 跨块 fast-weight 检索：只用先前累积（st>0 或已有外部 memories）
            if st > 0 or memories is not None:
                if bf16:
                    out[:, :, sl] = a + torch.einsum('bhwd,bhde->bhwe', q_c, new_mem)
                else:
                    out[:, :, sl] = a.float() + torch.einsum(
                        'bhwd,bhde->bhwe', q_c.float(), new_mem.float())
            else:
                out[:, :, sl] = a

            # ③ 累积本块 fast-weight（顺序与原实现一致：检索在前、累积在后）
            if bf16:
                new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c, v_c)
            else:
                new_mem = new_mem + torch.einsum(
                    'bhwd,bhwe->bhde', k_c.float(), v_c.float())

        return out, new_mem


def to_opt(model, strict_bf16=True):
    """原地把 model 每个 block 的 attn 换成融合实现。参数完全不变（只换类）。"""
    n = 0
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt
        blk.attn.strict_bf16 = strict_bf16
        n += 1
    assert n > 0, '没有找到 blocks'
    return model


class FWAttentionOpt2(FWAttention):
    """并行前缀和实现 —— 彻底消除 Python chunk 循环（方案 A 的正解）。

    关键洞察：原顺序循环是**前缀和结构**
        第 c 块检索用的 memory = Σ_{c'<c} K_{c'}^T V_{c'}
    这种东西不需要顺序累加，可以用 exclusive cumsum 一次性并行算出来
    （GLA / DeltaNet 的 chunked 实现正是这么做的）。

    数学与基线完全一致：
      块内: out_c  = softmax_causal(Q_c K_cᵀ) V_c          ← 批量 SDPA，1 次调用覆盖所有块
      跨块: M_c    = memories + Σ_{c'<c} K_{c'}^T V_{c'}    ← exclusive cumsum，并行
            out_c += Q_c @ M_c                              ← 1 次批量 einsum
      返回: new_mem = memories + Σ_{∀c} K_c^T V_c
    唯一差别是**求和顺序**（cumsum 树形归约 vs 顺序累加）→ 数值差在浮点噪声级。

    附带好处：不再需要 `out_chunks` 列表 → 显存不再随块数线性膨胀
    （原实现在 T=32768 会 OOM，本实现每层只需一个 [B,nch,nh,N,D] 的张量）。
    """

    strict_bf16 = True

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            return super().forward(Q, K, V, memories, W)   # 兜底：走顺序版
        nch = T // W

        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        q = QR.view(B, nh, nch, W, N)                       # k 与 q 同一份
        v = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D)

        # ① 块内因果注意：所有块一次批量 SDPA
        #    ⚠️⚠️ 必须传 **4-D**（把 nch 折进 batch 维）—— 实测 5-D 输入会让
        #    FLASH/MEMEFF/CUDNN 全部报 "No available kernel" 而**静默回退到 MATH 后端**，
        #    那就在做 fp32 bmm + 大量 direct_copy（profile 实测占 20% 是 sgemm + 21% 是 memcpy）。
        #    折成 4-D 后命中 MEMEFF 融合内核。v 的展开物化一次，代价远小于回退 MATH。
        qf = q.reshape(B * nh * nch, W, N)
        vf = v.reshape(B * nh * nch, W, D)
        a = F.scaled_dot_product_attention(
            qf, qf, vf, is_causal=True, scale=1.0).view(B, nh, nch, W, D)

        # ② 每块的 Σ_t k⊗v
        kv = torch.einsum('bhcwd,bhcwe->bchde', q, v)       # [B,nch,nh,N,D]

        # ③ exclusive 前缀和（第 c 块只用前 c 块）—— 并行替代顺序循环
        S = kv.cumsum(dim=1)
        S = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
        if memories is not None:
            S = S + memories.unsqueeze(1)

        # ④ 跨块检索（1 次批量 einsum）
        out = (a + torch.einsum('bhcwd,bchde->bhcwe', q, S)).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=1)
        if memories is not None:
            new_mem = new_mem + memories
        return out, new_mem


def to_opt2(model, strict_bf16=True):
    """原地换成并行前缀和实现。参数完全不变。"""
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt2
        blk.attn.strict_bf16 = strict_bf16
    return model


# ---------------------------------------------------------------------------
# 第三处根因：nn.LayerNorm 在 CUDA autocast 下**输出 fp32**（实测，有无 affine 都一样）。
# 我们 block 里有 4 个 LN → 每个 LN 之后的 encoder/encoder_v/decoder 全掉进 fp32 GEMM。
# profile 实测：ampere_sgemm × 36 次 = 19% 的 GPU 时间。
# 修法：LN 之后把激活 .to(输入 dtype) 拉回 bf16（梯度仍走 autocast 的正常路径）。
# ---------------------------------------------------------------------------
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycle, BDHBlockFWCycleLM


class BDHBlockFWCycleBF16(BDHBlockFWCycle):
    """与 BDHBlockFWCycle 数学一致，只把 LN 输出拉回 bf16，避免全链 fp32 GEMM。"""

    def forward(self, x, memories=None):
        C = self.config
        B = x.shape[0]
        T = x.shape[2]
        D = self.D
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        dt = x.dtype
        x = self.ln(x).to(dt)
        for _ in range(self.steps):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x,
                                     memories=memories, W=self.W)
            yKV = self.ln(yKV).to(dt)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = self.drop(x_sparse * y_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP).to(dt)
            x = self.ln(x + y).to(dt)
            memories = new_mem
        return x, memories


class BDHBlockFWCycleLMBF16(BDHBlockFWCycleLM):
    """LM 层：block 换成 BF16 版 + LM 自己的 ln 也拉回 bf16。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for blk in self.blocks:
            blk.__class__ = BDHBlockFWCycleBF16

    def forward(self, x, targets=None):
        B, T = x.size()
        h = self.e(x).unsqueeze(1)
        h = self.ln(h).to(h.dtype)
        for blk in self.blocks:
            h, _ = blk(h, None)
        lg = self.head(h.view(B, T, self.D))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, self.vocab), targets.view(-1))
        return lg, loss


def to_opt3(model, strict_bf16=True):
    """Opt2 + LN 后拉回 bf16。参数完全不变。"""
    for blk in model.blocks:
        blk.__class__ = BDHBlockFWCycleBF16
        blk.attn.__class__ = FWAttentionOpt2
        blk.attn.strict_bf16 = strict_bf16
    return model


def build_opt3(**kw):
    """直接建 Opt3（LM 类就位，无需先建基线再换）。"""
    return BDHBlockFWCycleLMBF16(**kw)


# ---------------------------------------------------------------------------
# OPT5：**不用 SDPA**。
# 根因（实测）：本架构 Q/K 在 N 维空间、V 在 D 维空间（N≠D），而 FlashAttention
# 要求 q/k/v 最后一维相同 → Flash 一律不可用 → SDPA 静默回退 MEMEFF/MATH，
# 而 MEMEFF 内部用 **fp32** 做 qkᵀ 与 p@v → profile 实测 36 次 ampere_sgemm = 25.5ms(9.5%)。
# 修法：手写块内因果注意走 bf16 张量核，语义与基线逐字对齐
#       （tril(diag=0) + -inf + softmax(fp32) + @v）。
# 同时把 [B,nh,nch,...] 折成 batch 维（bf=nh*nch），4 个 bmm 全部显式、无隐式 permute 拷贝。
# ---------------------------------------------------------------------------
class FWAttentionOpt5(FWAttention):
    """手写块内注意 + 并行前缀和跨块检索。参数与基线完全一致。"""

    strict_bf16 = True
    # True: 前缀和也在 bf16（更省带宽，数值略糙）；False: 前缀和留 fp32（贴近基线顺序累加）
    bf16_prefix = False
    # True: softmax 在 bf16 算（与 compile/Flash 内部一致，省两次 33.5M 元素往返 cast）
    # False: 严格对齐基线（基线是 softmax(sim.float())）
    bf16_softmax = False

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            return super().forward(Q, K, V, memories, W)
        nch = T // W
        bf_ = B * nh * nch

        r = torch.arange(0, T, device=self.freqs.device, dtype=torch.float32).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        qq = QR.reshape(bf_, W, N)                                  # 视图，无拷贝
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(bf_, W, D)

        # ① 块内因果注意（手写 bf16 bmm；基线语义：tril(0)+-inf+softmax(fp32)）
        sc = torch.bmm(qq, qq.transpose(1, 2))                      # [bf_, W, W]
        causal = torch.tril(torch.ones(W, W, device=Q.device, dtype=torch.bool), 0)
        sc = sc.masked_fill(~causal, float('-inf'))
        p = torch.softmax(sc if self.bf16_softmax else sc.float(), dim=-1)
        a = torch.bmm(p, vv)                                        # [bf_, W, D]

        # ② 每块的 Σ_w k⊗v = qᵀ v
        kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)

        # ③ exclusive 前缀和（S - kv 即 exclusive，省掉 cat/zeros 的全量拷贝）
        dt_sum = kv.dtype if self.bf16_prefix else torch.float32
        S = kv.cumsum(dim=2, dtype=dt_sum) - kv
        if memories is not None:
            S = S + memories.unsqueeze(2)

        # ④ 跨块检索（显式 bmm）
        o = torch.bmm(qq, S.reshape(bf_, N, D))
        out = (a + o).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=2, dtype=torch.float32)
        if memories is not None:
            new_mem = new_mem + memories.float()
        return out, new_mem


def to_opt5(model, strict_bf16=True, bf16_prefix=False, bf16_softmax=False):
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt5
        blk.attn.strict_bf16 = strict_bf16
        blk.attn.bf16_prefix = bf16_prefix
        blk.attn.bf16_softmax = bf16_softmax
    return model


# ---------------------------------------------------------------------------
# OPT6 / OPT7：**把 FlashAttention 从「天生不适用」变成「能用」**
#
# 根因回顾：本架构 Q/K 在 N=128 维、V 在 D=256 维 → Flash 要求等维 → 不可用
#           → SDPA 回退 MEMEFF/MATH 的 fp32 sgemm（OPT5 已用手写 bmm 绕开，但仍物化 [W,W] 分数矩阵）。
#
# 关键洞察：块内注意 softmax 权重只由 q·k 决定，**与 v 的列无关**，因此
#     softmax(QKᵀ)·V  ≡  concat_c[ softmax(QKᵀ)·V[:, c·N:(c+1)·N] ]
# 于是把 V 按列切成 D/N 份，每份与 Q/K 同维(N) → 两个 kernel 都满足 Flash 等维要求，**数学逐位精确**。
# 代价：scores 算 D/N 遍（+33% FLOPs），换来不物化 [nf,W,W] 分数矩阵 + 全程融合。
#
# OPT7 是另一条路：把 Q/K 零填充到 D 维（点积补 0 不变）→ 单次 flash，但 scores FLOPs ×2。
# ---------------------------------------------------------------------------
class FWAttentionOpt6(FWAttention):
    """块内注意走 FlashAttention（v 按列切份）。数学与基线逐位等价。"""

    strict_bf16 = True
    bf16_prefix = False
    force_backend = None       # None / 'FLASH' / 'MEM_EFFICIENT' / 'MATH'（探针用）

    def _sdpa(self, q, v):
        kw = dict(is_causal=True, scale=1.0)
        if self.force_backend == 'MATH':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, q, v, **kw)
        return F.scaled_dot_product_attention(q, q, v, **kw)

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0 or D % N != 0:
            return super().forward(Q, K, V, memories, W)
        nch = T // W
        nf = B * nh * nch
        ns = D // N

        r = torch.arange(0, T, device=self.freqs.device, dtype=torch.float32).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        qq = QR.reshape(nf, W, N)
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(nf, W, D)

        # ① 块内因果注意：v 按列切 ns 份，每份与 Q/K 同维 → 命中 Flash
        if ns == 1:
            a = self._sdpa(qq, vv)
        else:
            parts = vv.reshape(nf, W, ns, N).unbind(dim=2)
            a = torch.cat([self._sdpa(qq, p) for p in parts], dim=-1)

        # ② 跨块：并行 exclusive 前缀和（与 OPT5 同）
        kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)
        dt_sum = kv.dtype if self.bf16_prefix else torch.float32
        S = kv.cumsum(dim=2, dtype=dt_sum) - kv
        if memories is not None:
            S = S + memories.unsqueeze(2)
        o = torch.bmm(qq, S.reshape(nf, N, D))
        out = (a + o).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=2, dtype=torch.float32)
        if memories is not None:
            new_mem = new_mem + memories.float()
        return out, new_mem


class FWAttentionOpt7(FWAttention):
    """块内注意走 FlashAttention（Q/K 零填充到 D 维）。数学逐位等价，但 scores FLOPs ×2。"""

    strict_bf16 = True
    bf16_prefix = False

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0 or N > D:
            return super().forward(Q, K, V, memories, W)
        nch = T // W
        nf = B * nh * nch

        r = torch.arange(0, T, device=self.freqs.device, dtype=torch.float32).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        qq = QR.reshape(nf, W, N)
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(nf, W, D)

        # ① Q/K 零填充到 D 维（点积补 0 不改变 scores）→ 单次 flash
        if N < D:
            qp = F.pad(qq, (0, D - N))
        else:
            qp = qq
        a = F.scaled_dot_product_attention(qp, qp, vv, is_causal=True, scale=1.0)

        # ② 跨块：并行 exclusive 前缀和
        kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)
        dt_sum = kv.dtype if self.bf16_prefix else torch.float32
        S = kv.cumsum(dim=2, dtype=dt_sum) - kv
        if memories is not None:
            S = S + memories.unsqueeze(2)
        o = torch.bmm(qq, S.reshape(nf, N, D))
        out = (a + o).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=2, dtype=torch.float32)
        if memories is not None:
            new_mem = new_mem + memories.float()
        return out, new_mem


def to_opt6(model, strict_bf16=True, bf16_prefix=False):
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt6
        blk.attn.strict_bf16 = strict_bf16
        blk.attn.bf16_prefix = bf16_prefix
    return model


def to_opt7(model, strict_bf16=True, bf16_prefix=False):
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt7
        blk.attn.strict_bf16 = strict_bf16
        blk.attn.bf16_prefix = bf16_prefix
    return model


# ---------------------------------------------------------------------------
# OPT5-RAW：OPT5 骨架（手写 bmm + 并行 exclusive 前缀和）+ **块内 raw 读侧**。
#
# 背景（2026-09-12）：v6 基线把块内读侧从 softmax 改为 raw（能力 +8.5 分，8 seed 统计确认），
#   但 OPT1/2/6/7 优化的是 **softmax 语义**（SDPA / FlashAttention 内核内部强制 softmax），
#   OPT5 也是手写 softmax → **现有全部优化都无法表达 raw**。raw 与 SDPA 互斥。
# 因此 raw 的最优形态只能：手写 bmm（材料化 [W,W] 分数矩阵）+ 前缀和，但省掉 softmax 的
#   exp + 归约（raw 只做 masked_fill(0) + bmm）→ 理论上比 OPT5 的 softmax 版更快。
#
# raw 语义（严格对齐 fused_fw_fw_cycle.FWAttention read_mode='raw'）：
#   sim = q·kᵀ ; mask = tril(diag=-1) → 置 **0**（不是 -inf）；**不做 softmax**；out = sim·v
# ---------------------------------------------------------------------------
class FWAttentionOpt5Raw(FWAttention):
    """OPT5 骨架 + raw 块内读侧。参数与基线完全一致（只换实现形态）。"""

    strict_bf16 = True
    # True: 前缀和也在 bf16（省带宽）; False: 前缀和留 fp32（贴近基线顺序累加）
    bf16_prefix = False
    # 缓存的 tril(diag=-1) 掩码（W 固定 → 只建一次，省掉每块 fill_）
    _mask_cache = None

    @classmethod
    def _tril_mask(cls, W, device):
        c = cls._mask_cache
        if c is None or c[0] != W or c[1] != str(device):
            m = torch.tril(torch.ones(W, W, device=device, dtype=torch.bool), diagonal=-1)
            c = (W, str(device), m)
            cls._mask_cache = c
        return c[2]

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            return super().forward(Q, K, V, memories, W)
        nch = T // W
        bf_ = B * nh * nch

        r = torch.arange(0, T, device=self.freqs.device, dtype=torch.float32).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        qq = QR.reshape(bf_, W, N)
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(bf_, W, D)

        # ① 块内 raw 因果注意（手写 bf16 bmm；无 softmax，mask=0 真 0，diagonal=-1 不看自己）
        sc = torch.bmm(qq, qq.transpose(1, 2))                       # [bf_, W, W]
        mask = self._tril_mask(W, Q.device)
        sc = sc.masked_fill(~mask, 0.0)
        a = torch.bmm(sc, vv)                                        # [bf_, W, D]

        # ② 每块的 Σ_w k⊗v = qᵀ v
        kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)

        # ③ exclusive 前缀和（与 OPT5 同：S - kv 即 exclusive）
        dt_sum = kv.dtype if self.bf16_prefix else torch.float32
        S = kv.cumsum(dim=2, dtype=dt_sum) - kv
        if memories is not None:
            S = S + memories.unsqueeze(2)

        # ④ 跨块检索（显式 bmm）
        o = torch.bmm(qq, S.reshape(bf_, N, D))
        out = (a + o).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=2, dtype=torch.float32)
        if memories is not None:
            new_mem = new_mem + memories.float()
        return out, new_mem


def to_opt5_raw(model, strict_bf16=True, bf16_prefix=False):
    """OPT5 骨架 + raw 块内读侧（与 v6 新默认 read_mode='raw' 语义对齐）。"""
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionOpt5Raw
        blk.attn.strict_bf16 = strict_bf16
        blk.attn.bf16_prefix = bf16_prefix
    return model


# ---------------------------------------------------------------------------
# B 线：块内注意走 **TileLang raw kernel**（分数矩阵不落显存）+ 跨块 exclusive 前缀和。
#
# 孤立段实测（B=2 nh=8 w=256 N=128 D=256，bf16，5×50 中位）：
#   A 线交付 compile(物化展开) 105.78 ms   vs   TileLang kernel 15.44 ms  → 6.85×
#   （torch.compile 对纯 bmm 段无效：105.78 vs eager 105.31；其收益只在整模型的小算子融合）
# ⚠️ kernel 的 S 维度是编译期常量 → 每个新 T 需重新编译（约 6s），测速时须排除编译时间。
# ---------------------------------------------------------------------------
class FWAttentionTLRaw(FWAttention):
    """B 线实现：TileLang 块内 + exclusive 前缀和跨块。参数与基线完全一致。"""

    strict_bf16 = True
    bf16_prefix = False
    block_M = 64
    block_N = 64
    threads = 128
    num_stages = 1
    _kern_cache = {}

    @classmethod
    def _kernel(cls, B, nh, T, N, D_):
        key = (B, nh, T, N, D_, cls.block_M, cls.block_N, cls.threads, cls.num_stages)
        if key not in cls._kern_cache:
            import sys as _sys
            if '/data/dynfw' not in _sys.path:
                _sys.path.insert(0, '/data/dynfw')
            from benchmarks.tl_attn_raw import build_tilelang_raw
            cls._kern_cache[key] = build_tilelang_raw(
                B, nh, T, N, D_, 'bf16', cls.block_M, cls.block_N, cls.threads, cls.num_stages)
        return cls._kern_cache[key]

    def forward(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            return super().forward(Q, K, V, memories, W)
        nch = T // W
        bf_ = B * nh * nch

        r = torch.arange(0, T, device=self.freqs.device, dtype=torch.float32).view(1, 1, -1, 1)
        QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        # ① 块内 raw 因果注意：TileLang kernel
        #    ⚠️ 语义边界：v6 的块内注意是【只在块内】(窗口 = W)，跨块信息必须走 fast-weight。
        #    因此把 [B,nh,nch,W,N] 折成 batch 维 (B*nch) 逐块独立做因果，
        #    **不能**把整条序列一起喂进去（那会变成全局因果 = 偷读跨块）。
        # ⚠️ batch 维顺序必须与后面 qq/vv 的 reshape 一致：(b, h, c)，heads 折成 1。
        #    q5 = QR.reshape -> 纯视图零拷贝；只有 V 的 nh 展开物化一次（与 OPT5Raw 同）。
        kern = self._kernel(bf_, 1, W, N, D)
        q5 = QR.reshape(bf_, 1, W, N)
        v5 = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(bf_, 1, W, D)
        if q5.dtype != torch.bfloat16:
            q5 = q5.to(torch.bfloat16)
        if v5.dtype != torch.bfloat16:
            v5 = v5.to(torch.bfloat16)
        a = kern(q5, q5, v5).reshape(bf_, W, D).to(QR.dtype)     # [bf_, W, D] 对齐 o 的形状

        # ② 每块 Σ k⊗v（bf16 bmm，对齐 OPT5Raw）
        qq = QR.reshape(bf_, W, N)
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(bf_, W, D)
        kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)

        # ③ exclusive 前缀和
        dt_sum = kv.dtype if self.bf16_prefix else torch.float32
        S = kv.cumsum(dim=2, dtype=dt_sum) - kv
        if memories is not None:
            S = S + memories.unsqueeze(2)

        # ④ 跨块检索
        o = torch.bmm(qq, S.reshape(bf_, N, D))
        out = (a + o).reshape(B, nh, T, D)

        new_mem = kv.sum(dim=2, dtype=torch.float32)
        if memories is not None:
            new_mem = new_mem + memories.float()
        return out, new_mem


def to_tl_raw(model, **kw):
    """原地换成 TileLang raw 实现。参数完全不变。"""
    for blk in model.blocks:
        blk.attn.__class__ = FWAttentionTLRaw
        for k, v in kw.items():
            setattr(blk.attn, k, v)
    return model
