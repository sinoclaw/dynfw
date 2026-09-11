"""DynFW v8: DLA 状态槽 + MoBA 式【槽选择】—— v7 读侧无差别 sum 的对照修复版。

════════════════════════════════════════════════════════════════════════
参照(全部已核实原文, 不凭空造)
════════════════════════════════════════════════════════════════════════
- DLA  arXiv:2606.10650 (2026-06-09) —— v7 的参照。多状态线性注意 + 信息感知动态
  状态合并 + 容量有界(chronologically ordered)状态缓存。其读侧 o_t = Σ_i φ(q)·S_i,
  即【对所有状态槽无差别求和】。
- MoBA arXiv:2502.13189 (2025-02-18, Moonshot) —— 把 MoE 原则用于 attention: 每个
  query 对分块做【门控 Top-K 选择】, 打分 s_i = <q, mean(K_block_i)>; "less structure"
  原则让模型自主决定 attend 哪里。块级选择 = query-dependent 的稀疏。
- NSA  arXiv:2502.11089 (2025-02-16, DeepSeek) —— 动态分层稀疏: 粗粒度 token 压缩 +
  细粒度 token 选择。是 DeepSeek-V4.1 CSA2 的祖先。

════════════════════════════════════════════════════════════════════════
问题定位 (对照 DeepSeek-V4.1-Flash 技术报告后的发现)
════════════════════════════════════════════════════════════════════════
V4.1 CSA2 的 Reuse Mode = 跨层复用 **Top-K 索引**(选择性读取), 而非无差别聚合;
论文原话: "For a fixed candidate-pool size, this changes the per-query cost of deeper
indexers from linear in context length to constant."

对照我们 v7 (fused_fw_dla_cycle.py 第 137 行):
    S_summed = S[:, :, :used, :, :].sum(dim=2)        # ← 所有已用槽【无差别求和】
    retr = einsum('bhwd,bhde->bhwe', q, S_summed)     # ← 然后才检索
**求和先发生 → 状态槽的区分度被提前抹掉** —— 无论 query 是什么, 读到的都是同一个
"槽总和"。这与 MoBA/CSA2 的"按相关性选择性读取"正相反, 且 v7 实测能力垫底
(corpus_en 3-seed 101.22)。

⚠️ 诚实: 无差别 sum【严格对齐 DLA 原文】(原文就是 Σ_i)。所以本版不是"修 bug",
而是**把 MoBA 的选择机制嫁接到 DLA 的多状态槽上**——属新机制, 须按军规走
"提出→参照→载体判析→照参照实现→多 seed 验证", 并如实标注复杂度/来源。

════════════════════════════════════════════════════════════════════════
载体判析 (MoBA 的机制前提, 我们的载体满足吗?)
════════════════════════════════════════════════════════════════════════
MoBA 门控需要: 每个块有一个可用于打分的"代表" (它用 mean(K_block))。
我们的载体: DLA 的槽 S_j = Σ_t (k_t ⊗ v_t) ∈ [N,D] 是 **k⊗v 的混合**, 无法从中
          反解出 key 侧统计量 → 直接照搬 <q, mean(K_block)> 不可行。
补前提(不改参数, 只加**状态**): 槽同时累积 key 侧和 `ksum_j = Σ_t k_t ∈ [N]`。
    → 打分 s_j = <q, ksum_j> = Σ_t (q·k_t)   ← 与 MoBA 的 <q, mean(K_block)> 同形
       (MoBA 取 mean, 我们取 sum; 二者只差常数因子, 对 softmax 无影响)
    → ksum 是状态(从 k 派生)不是参数 → **不破坏"同参数"对轰口径** ✓

════════════════════════════════════════════════════════════════════════
⚠️ 归一化红线 (军规: 凡是非论文原文的归一化/缩放/除法/门控, 默认打问号)
════════════════════════════════════════════════════════════════════════
原版读侧尺度 = Σ_j o_j  (满 K 槽时 ≈ K·E[o])
softmax 版     = Σ_j α_j o_j，Σα=1 (尺度 ≈ E[o]) → **尺度差 K 倍!**
前车之鉴: v7 曾擅自加 `retr/max(used,1)`, K=4 时 KL 劣化到 112.04。
故本版把"选择"与"尺度"拆成两个正交开关, 逐一对照(一次只动一组变量):
    read_mode = 'sum'       : retr = Σ_j o_j                  (原版 baseline, 尺度~K)
    read_mode = 'softmax'   : α=softmax(s);  Σ α_j o_j        (选择,  尺度~1)
    read_mode = 'softmaxK'  : α=softmax(s)*K;Σ α_j o_j        (选择,  尺度~K ← 对齐原版)
    read_mode = 'topk'      : 硬选 k 槽 + softmax(s_sel)*k    (真稀疏, 尺度~k)
另设 force_uniform=True 强制 α≡1/K → 数学上与原版差常数 K, 用作实现正确性探针。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class DLAStateCacheTopK(nn.Module):
    """DLA 状态槽缓存(容量 K 有界, 信息感知合并) + key 侧和(k 用于 MoBA 式打分)。

    state = (S: [B,nh,K,N,D] 状态, I: [B,nh,K,1] 信息分, n: [B,nh,K,1] token 计数,
             ksum: [B,nh,K,N] key 侧和)
    """

    def __init__(self, config, K=16):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.nh = nh
        self.N = N
        self.D = D
        self.K = K

    def info_score(self, block_rep):
        """DLA 信息分: 块内 token 相对块首的表示变化量(L2) 求和。"""
        head = block_rep[:, :, :1, :]
        diff = (block_rep - head).norm(dim=-1, keepdim=True)
        return diff.sum(dim=2)  # [B,nh,1]

    def merge_lowest_density(self, S, I, n, Ksum):
        """DLA Algorithm 2: 合并相邻信息密度最低对 Ī_i+Ī_i+1 / (n_i+n_i+1)。"""
        denom = n[:, :, :-1, :] + n[:, :, 1:, :] + 1e-8
        dens = (I[:, :, :-1, :] + I[:, :, 1:, :]) / denom
        if dens.shape[2] == 0:
            return S, I, n, Ksum
        idx = dens.argmin(dim=2)  # [B,nh,1]
        S2, I2, n2, K2 = S.clone(), I.clone(), n.clone(), Ksum.clone()
        for b in range(S.shape[0]):
            for h in range(S.shape[1]):
                i0 = int(idx[b, h, 0].item())
                S2[b, h, i0] = S[b, h, i0] + S[b, h, i0 + 1]
                I2[b, h, i0] = I[b, h, i0] + I[b, h, i0 + 1]
                n2[b, h, i0] = n[b, h, i0] + n[b, h, i0 + 1]
                K2[b, h, i0] = Ksum[b, h, i0] + Ksum[b, h, i0 + 1]   # ← key 侧和同步合并
        keep = torch.ones_like(n2[:, :, :, 0], dtype=torch.bool)
        for b in range(S.shape[0]):
            for h in range(S.shape[1]):
                keep[b, h, int(idx[b, h, 0].item()) + 1] = False
        kk = keep.unsqueeze(-1).unsqueeze(-1)
        newS = S2[kk.expand_as(S2)].view(S.shape[0], S.shape[1], S.shape[2] - 1, S.shape[3], S.shape[4])
        newI = I2[keep.unsqueeze(-1).expand_as(I2)].view(I.shape[0], I.shape[1], I.shape[2] - 1, I.shape[3])
        newn = n2[keep.unsqueeze(-1).expand_as(n2)].view(n.shape[0], n.shape[1], n.shape[2] - 1, n.shape[3])
        newK = K2[keep.unsqueeze(-1).expand_as(K2)].view(Ksum.shape[0], Ksum.shape[1], Ksum.shape[2] - 1, Ksum.shape[3])
        return newS, newI, newn, newK


class DLASlotAttn(nn.Module):
    """块内精确注意(窗口 W) + DLA 状态槽 + 【MoBA 式槽选择读侧】(唯一改动点)。"""

    def __init__(self, config, K=16, read_mode="softmaxK", topk=4, temp=1.0):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(get_freqs(N, theta=2 ** 16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh, self.N, self.D = nh, N, D
        self.cache = DLAStateCacheTopK(config, K=K)
        self.read_mode = read_mode
        self.topk = topk
        self.temp = temp
        self.force_uniform = False   # 探针开关: 强制 α≡1/K (实现正确性验证用)
        self.capture_retr = False    # 诊断开关: True 时把每块的跨槽检索结果存进 last_retr
        self.last_retr = []

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = DLASlotAttn.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N](K is Q); V:[B,1,T,D]; memories: 状态槽缓存元组。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N, D, Kcap = self.N, self.D, self.cache.K
        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)
        KR = QR
        out_chunks = []
        if self.capture_retr:
            self.last_retr = []
        if memories is None:
            S = torch.zeros(B, nh, Kcap, N, D, device=Q.device)
            I = torch.zeros(B, nh, Kcap, 1, device=Q.device)
            n = torch.zeros(B, nh, Kcap, 1, device=Q.device)
            Ksum = torch.zeros(B, nh, Kcap, N, device=Q.device)
            used = 0
        else:
            S, I, n, Ksum, used = memories

        for st in range(0, T, W):
            en = min(st + W, T)
            q_c = QR[:, :, st:en]
            k_c = KR[:, :, st:en]
            v_c = V[:, :, st:en]
            w = en - st
            # ---- 块内精确注意(与 v6/v7 完全一致) ----
            sim = q_c @ k_c.mT
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            sim = sim.masked_fill(~causal, float('-inf'))
            attn = torch.softmax(sim.float(), dim=-1)
            agg = attn @ v_c                      # [B,nh,w,D]

            # ---- 跨槽检索(★唯一改动: MoBA 式选择, 替代 v7 的无差别 sum) ----
            if used > 0:
                # ⚠️ .clone() 必须: S[:,:,:used] 只是【视图】, 而本块末尾会原地写
                #   `S[:, :, used] = slot_S` → 若不 clone, autograd 报
                #   "variable needed for gradient computation has been modified by an
                #    inplace operation" (实测 nh=16 后端炸)。v7 原版因用 .sum() 产生新张量
                #   侥幸规避; 选择性读直接对 S_u 做 einsum, 必须显式物化。
                S_u = S[:, :, :used].float().clone()       # [B,nh,u,N,D]
                Ks_u = Ksum[:, :, :used].float().clone()   # [B,nh,u,N]
                qf = q_c.float()                   # [B,nh,w,N]
                if self.read_mode == "sum":
                    # 原版 v7 语义: 先 sum 槽再检索 (保留以做严格对照)
                    retr = torch.einsum('bhwd,bhde->bhwe', qf, S_u.sum(dim=2))
                else:
                    # MoBA 式门控: s_j = <q, ksum_j>  (= Σ_t q·k_t, 与 MoBA <q,mean(K_b)> 同形)
                    s = torch.einsum('bhwd,bhud->bhwu', qf, Ks_u) / self.temp   # [B,nh,w,u]
                    if self.force_uniform:
                        alpha = torch.full_like(s, 1.0 / used)                  # 探针: α≡1/K
                    elif self.read_mode == "softmax":
                        alpha = torch.softmax(s, dim=-1)                        # 尺度 ~1
                    elif self.read_mode == "softmaxK":
                        alpha = torch.softmax(s, dim=-1) * used                 # 尺度 ~K 对齐原版
                    elif self.read_mode == "topk":
                        tk = min(self.topk, used)
                        val, idx = s.topk(tk, dim=-1)
                        alpha = torch.zeros_like(s).scatter(-1, idx, torch.softmax(val, dim=-1) * tk)
                    else:
                        raise ValueError(f"unknown read_mode {self.read_mode}")
                    # retr = Σ_j α_j (q @ S_j)
                    #   ⚠️ 必须【先对每个槽算 q@S_j, 再按 α 加权】——
                    #   曾误写 einsum('bhwu,bhude->bhwe', α, S_u) (把 q 丢了: u,d 全被缩并
                    #   = 对 S 做列和, 与 q 无关), P2 尺度探针当场抓到。正确分两步:
                    O = torch.einsum('bhwd,bhude->bhwue', qf, S_u)   # [B,nh,w,u,D] 每槽检索
                    retr = torch.einsum('bhwu,bhwue->bhwe', alpha, O)  # [B,nh,w,D] α 加权
                agg = agg + retr.to(agg.dtype)
                if self.capture_retr:
                    self.last_retr.append(retr.detach().clone())
            out_chunks.append(agg)

            # ---- 状态槽更新(与 v7 完全一致, 仅 ksum 同步维护) ----
            info = self.cache.info_score(k_c.float())
            slot_S = torch.einsum('bhwd,bhwe->bhde', k_c.float(), v_c.float())
            slot_K = k_c.float().sum(dim=2)                                  # [B,nh,N] key 侧和
            if used < Kcap:
                S[:, :, used] = slot_S
                I[:, :, used] = info
                n[:, :, used] = torch.full((B, nh, 1), float(w), device=Q.device)
                Ksum[:, :, used] = slot_K
                used += 1
            else:
                # 满 → 严格 DLA: 先合并相邻最低密度对(K→K-1), 再【追加】本块(回到 K)
                #   ⚠️ 存量 bug(2026-09-10 探针 P3 抓出): 原写法 `S_m[:,:,-1]=slot_S` 是
                #   【覆盖】合并后的末槽 → 净效果槽数每合并一次减 1 (K→K-1→K-2→...),
                #   而 used 却硬写回 K → S[:,:,:used] 被 python 切片静默 clamp,
                #   记忆容量随上下文单调萎缩。DLA 原文要求 fixed-size chronologically
                #   ordered cache → 必须 cat 追加, 不能覆盖。
                S_m, I_m, n_m, K_m = self.cache.merge_lowest_density(S, I, n, Ksum)
                S = torch.cat([S_m, slot_S.unsqueeze(2)], dim=2)
                I = torch.cat([I_m, info.unsqueeze(2)], dim=2)
                n = torch.cat([n_m, torch.full((B, nh, 1, 1), float(w), device=Q.device)], dim=2)
                Ksum = torch.cat([K_m, slot_K.unsqueeze(2)], dim=2)
                used = Kcap
        out = torch.cat(out_chunks, dim=2)
        return out, (S, I, n, Ksum, used)


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult
        self.vocab_size = vocab


class BDHBlockSlotCycle(nn.Module):
    """v8: 与 v7 BDHBlockDLACycle 完全一致, 唯一差异 = attn 读侧用 MoBA 式槽选择。"""

    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512, K=16,
                 read_mode="softmaxK", topk=4):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg
        self.D = D
        self.vocab = vocab
        self.steps = steps
        self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = DLASlotAttn(cfg, K=K, read_mode=read_mode, topk=topk)
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

    def forward(self, x, memories=None):
        C = self.config
        B = x.shape[0]
        T = x.shape[2]
        D = self.D
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.ln(x)
        for _ in range(self.steps):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x, memories=memories, W=self.W)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
            memories = new_mem
        return x, memories

    def np(self):
        return sum(p.numel() for p in self.parameters())


class BDHBlockSlotCycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockSlotCycle -> head。"""

    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128,
                 W=512, K=16, read_mode="softmaxK", topk=4):
        super().__init__()
        self.D, self.nh, self.vocab = D, nh, vocab
        self.n_layer, self.steps, self.W = n_layer, steps, W
        self.read_mode = read_mode
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([
            BDHBlockSlotCycle(D, nh, mlp_mult, vocab, steps=steps, W=W, K=K,
                              read_mode=read_mode, topk=topk)
            for _ in range(n_layer)])
        self.head = nn.Linear(D, vocab, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        h = self.e(x).unsqueeze(1)
        h = self.ln(h)
        # 修复 2026-09-11：原 `mem = None; for blk: h, mem = blk(h, mem)` 造成未来泄漏 ——
        # FWAttention 返回的 new_mem 是【整个序列】的 k⊗v 累积，下一 block 在位置 t 检索时
        # 读到 t 之后的 k⊗v。逐层 hook 实测 block0 因果 OK、block1 起发散(2.3e-2→14.3)。
        # 改为每 block 独立 memory（层内 chunk 间仍累积，因果正确）。探针验证 maxdiff=0。
        for blk in self.blocks:
            h, _ = blk(h, None)
        lg = self.head(h.view(B, T, self.D))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, self.vocab), targets.view(-1))
        return lg, loss

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
