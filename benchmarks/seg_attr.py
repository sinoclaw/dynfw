"""整模型 prefill 时间构成的【语义段】归因。

回答的问题（为「TileLang 全段化」排期提供依据）：
  我们 A″ 交付形态（to_opt5 + torch.compile）在 T=8192 / B=2 下，
  前向时间里各语义段各占几成？其中【块内注意】段占多少？

为什么需要它：
  块内注意单段微基准已实测 compile 112.0µs -> TileLang 24.0µs（4.667×，见 tl_attn.py），
  但整模型里这一段占多少**未知**。没有这个占比，任何「全段化能快多少」都是拍脑袋。
  本脚本给出占比 -> 天花板 = 1/(1 - share_intra*(1 - 24.0/112.0))

⚠️ 这是**诊断性消融**（ablation）：被消掉的段改成「错但不崩」的廉价实现，只测时间不测正确性。
   每个变体各自独立 torch.compile —— 同形态对比，遵守「消融必同口径」纪律。
   口径：**前向 only**（no_grad + autocast bf16），不是 fwd+bwd。别和 A″ 的 39.3ms/step 直接比。

用法: PYTHONPATH=/data/dynfw python benchmarks/seg_attr.py --T 8192 --batch 2
输出: /data/logs/seg_attr.json + 控制台表
"""
import sys
import json
import time
import types
import argparse

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import (          # noqa: E402
    BDHBlockFWCycleLM, BDHBlockFWCycle, FWAttention)
from dynfw.models.fused_fw_fw_cycle_opt import (      # noqa: E402
    to_opt5, FWAttentionOpt)

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
MM = 4                       # mlp_mult（--mm 覆盖）
FORM = 'compile'             # 'compile' | 'eager'（--form 覆盖）
BIG_T = 262144
# tl_attn.py 微基准（同形状 B=2,nh=8,w=256,N=128,D=256，CUDA events 中位）
T_MICRO_EAGER, T_MICRO_COMPILE, T_MICRO_TILELANG = 0.1426, 0.1120, 0.0240


# ---------------------------------------------------------------- 变体 forward
def make_attn_fwd(no_intra=False, no_cross=False, no_rope=False):
    """FWAttentionOpt5.forward 的消融副本（数学与原件逐行一致，仅按 flag 短路）。"""

    def fwd(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            # 本脚本所有配置都满足 T % W == 0（T=8192, W=256）；非整除情形不在消融范围内
            raise NotImplementedError(f'本消融要求 T % W == 0，实得 T={T} W={W}')
        nch = T // W
        bf_ = B * nh * nch

        if no_rope:
            QR = Q
        else:
            r = torch.arange(0, T, device=self.freqs.device,
                             dtype=torch.float32).view(1, 1, -1, 1)
            QR = FWAttentionOpt.rope_fast(r * self.freqs, Q)

        qq = QR.reshape(bf_, W, N)
        vv = V.view(B, 1, nch, W, D).expand(B, nh, nch, W, D).reshape(bf_, W, D)

        if no_intra:                                  # ① 块内注意 -> 廉价替身
            a = vv
        else:
            sc = torch.bmm(qq, qq.transpose(1, 2))
            causal = torch.tril(torch.ones(W, W, device=Q.device, dtype=torch.bool), 0)
            sc = sc.masked_fill(~causal, float('-inf'))
            p = torch.softmax(sc.float(), dim=-1)
            a = torch.bmm(p, vv)

        if no_cross:                                  # ②③④ 跨块路径 -> 跳过
            out = a.reshape(B, nh, T, D)
            new_mem = torch.zeros(B, nh, N, D, device=Q.device, dtype=torch.float32)
        else:
            kv = torch.bmm(qq.transpose(1, 2), vv).view(B, nh, nch, N, D)
            S = kv.cumsum(dim=2, dtype=torch.float32) - kv
            o = torch.bmm(qq, S.reshape(bf_, N, D))
            out = (a + o).reshape(B, nh, T, D)
            new_mem = kv.sum(dim=2, dtype=torch.float32)
        return out, new_mem

    return fwd


def make_block_fwd(no_mlp=False):
    """BDHBlockFWCycle.forward 的消融副本。"""

    def fwd(self, x, memories=None):
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
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x,
                                     memories=memories, W=self.W)
            yKV = self.ln(yKV)
            if no_mlp:                                # 跳过 encoder_v/relu/乘门控/decoder
                yMLP = x.reshape(B, 1, T, D)
            else:
                y_latent = yKV @ self.encoder_v
                y_sparse = F.relu(y_latent)
                xy_sparse = self.drop(x_sparse * y_sparse)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
            memories = new_mem
        return x, memories

    return fwd


def lm_fwd_nohead(self, x, targets=None):
    B, T = x.size()
    h = self.ln(self.e(x).unsqueeze(1))
    for blk in self.blocks:
        h, _ = blk(h, None)
    return h.view(B, T, self.D), None


def autocast_ctx():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def op_table(m, T, B, topn=22):
    """op/kernel 级 self CUDA 时间表。
    compile 形态下 kernel 名是 triton 融合名（不透明），但仍能区分
    cuBLAS GEMM 内核 与 融合 elementwise 内核 —— 这正是「有没有融合空间」的判据。"""
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    with torch.no_grad():
        with autocast_ctx():
            m(x, y)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p:
            with autocast_ctx():
                m(x, y)
            torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.key, e.count)
            for e in p.key_averages() if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    out = [(us / tot, us / 1000, k[:62], c) for us, k, c in rows[:topn]]
    return out, tot / 1000, len(rows)


FWAttentionOpt5_forward = None      # 延迟绑定（避免与类定义顺序纠缠）


def build(no_intra=False, no_cross=False, no_rope=False, no_mlp=False, no_head=False):
    from dynfw.models.fused_fw_fw_cycle_opt import FWAttentionOpt5
    torch.manual_seed(0)
    m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1,
                          mlp_mult=MM, W=W)
    to_opt5(m, True, False, False)
    global FWAttentionOpt5_forward
    FWAttentionOpt5_forward = FWAttentionOpt5.forward

    af = make_attn_fwd(no_intra, no_cross, no_rope)
    bf = make_block_fwd(no_mlp)
    for blk in m.blocks:
        blk.attn.forward = types.MethodType(af, blk.attn)
        blk.forward = types.MethodType(bf, blk)
    if no_head:
        m.forward = types.MethodType(lm_fwd_nohead, m)
    m = m.cuda()
    return torch.compile(m) if FORM == 'compile' else m


def bench(m, T, B, rounds=5, iters=3, warmup=2):
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                m(x, y)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        s = []
        for _ in range(rounds):
            ev[0].record()
            for _ in range(iters):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    m(x, y)
            ev[1].record()
            torch.cuda.synchronize()
            s.append(ev[0].elapsed_time(ev[1]) / iters)
    s.sort()
    return s[len(s) // 2], s[0]


def main():
    global VOCAB, D, NH, NL, W, MM, FORM
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--D', type=int, default=D)
    ap.add_argument('--nh', type=int, default=NH)
    ap.add_argument('--nl', type=int, default=NL)
    ap.add_argument('--mm', type=int, default=MM)
    ap.add_argument('--vocab', type=int, default=VOCAB)
    ap.add_argument('--w', type=int, default=W)
    ap.add_argument('--form', choices=['compile', 'eager'], default='compile')
    ap.add_argument('--out', default='/data/logs/seg_attr.json')
    a = ap.parse_args()
    T, B = a.T, a.batch
    VOCAB, D, NH, NL, W = a.vocab, a.D, a.nh, a.nl, a.w
    MM, FORM = a.mm, a.form

    variants = [
        ('全量(交付形态)',        dict()),
        ('-① 块内注意',           dict(no_intra=True)),
        ('-②③④ 跨块路径',        dict(no_cross=True)),
        ('-rope',                dict(no_rope=True)),
        ('-MLP 门控',             dict(no_mlp=True)),
        ('-输出头',               dict(no_head=True)),
    ]
    res = {}
    print('=' * 88)
    print(f'整模型 prefill 语义段归因   T={T}  B={B}  前向 only (no_grad + autocast bf16)')
    print(f'形态 = to_opt5 + {FORM}；config: D={D} nh={NH} nl={NL} mm={MM} '
          f'N={MM*D//NH} vocab={VOCAB} W={W}')
    print('   ⚠️ 段占比与 nl 无关（所有段都 ∝ 层数）→ 可用小 nl 复制大模型的每层构成')
    print('=' * 88)
    for name, kw in variants:
        try:
            m = build(**kw)
            med, mn = bench(m, T, B)
            full = res.get('全量(交付形态)')
            delta = (full[0] - med) if full else 0.0
            res[name] = (med, mn)
            print(f'{name:<18}{med:>9.2f} ms   (min {mn:7.2f})   '
                  f'{"Δ = " + format(delta, ".2f") + " ms" if full else "(基准)":>16}')
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            res[name] = None
            print(f'{name:<18}{"FAIL":>9}   {type(e).__name__}: {str(e)[:70]}')
            torch.cuda.empty_cache()

    json_out = {'T': T, 'B': B, 'form': f'to_opt5+{FORM}',
           'cfg': {'D': D, 'nh': NH, 'nl': NL, 'mm': MM, 'N': MM*D//NH,
                   'vocab': VOCAB, 'W': W},
           'mode': 'forward_only',
           'micro': {'eager': T_MICRO_EAGER, 'compile': T_MICRO_COMPILE,
                     'tilelang': T_MICRO_TILELANG},
           'results': {k: (None if v is None else {'median_ms': v[0], 'min_ms': v[1]})
                       for k, v in res.items()}}

    full = res.get('全量(交付形态)')
    if full:
        print()
        print('-' * 88)
        print('【块内注意占比】= 全段化决策的关键数')
        for key in ('-① 块内注意', '-②③④ 跨块路径', '-rope', '-MLP 门控', '-输出头'):
            v = res.get(key)
            if v:
                share = (full[0] - v[0]) / full[0]
                json_out.setdefault('shares', {})[key] = share
                print(f'  {key:<18} 占全量 {share:>6.1%}   ({full[0]-v[0]:.2f} ms)')
        vi = res.get('-① 块内注意')
        if vi:
            share = (full[0] - vi[0]) / full[0]
            gain = 1.0 - T_MICRO_TILELANG / T_MICRO_COMPILE
            cell = 1.0 / (1.0 - share * gain)
            json_out['share_intra'] = share
            json_out['tilelang_gain_per_call'] = gain
            json_out['ceiling_full_segment_tilelang'] = cell
            print()
            print(f'  ⇒ 块内注意占 {share:.1%}；该段【孤立微基准】可提速 '
                  f'{T_MICRO_COMPILE/T_MICRO_TILELANG:.2f}×（{T_MICRO_COMPILE*1000:.0f}µs→{T_MICRO_TILELANG*1000:.0f}µs）')
            print(f'  ⇒ 【全段化天花板 = {cell:.3f}×】(仅块内注意段，其余不动；⚠️ 乐观上界——见下)')
            print(f'     ⚠️ in-situ 每块代价 = {full[0]-vi[0]:.2f}ms / {NL*(T//W)} 块 = '
                  f'{(full[0]-vi[0])/ (NL*(T//W)) *1000:.1f}µs，而孤立 compile 是 {T_MICRO_COMPILE*1000:.1f}µs '
                  f'→ 交付形态里这段已被 compile 重度重叠，孤立微基准不可转移')
            print(f'     判据建议：≥1.30× 排期；1.15~1.30× 只做单段；<1.15× 划掉')
        print('-' * 88)

        print('\n【op 级 top22（按 self CUDA 时间）】')
        try:
            mr = build()
            tbl, gpu_ms, nk = op_table(mr, T, B)
            json_out['op_table'] = [{'pct': p, 'ms': ms, 'key': k, 'count': c}
                                    for p, ms, k, c in tbl]
            json_out['gpu_self_total_ms'] = gpu_ms
            json_out['n_kernel_keys'] = nk
            gemm = sum(ms for _, ms, k, _ in tbl if 'gemm' in k or 's16816' in k or 's1688' in k)
            tri = sum(ms for _, ms, k, _ in tbl if 'triton' in k)
            print(f'   GPU self total {gpu_ms:.1f} ms，kernel 种类 {nk} 个')
            print(f'   其中 cuBLAS GEMM 内核 {gemm:.2f} ms / triton 融合内核 {tri:.2f} ms')
            for p, ms, k, c in tbl:
                print(f'   {p:6.1%} {ms:8.2f}ms {c:>7}  {k}')
            del mr
            torch.cuda.empty_cache()
        except Exception as e:
            print('   op_table 失败:', type(e).__name__, str(e)[:120])

    with open(a.out, 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f'已写入 {a.out}')


if __name__ == '__main__':
    main()
