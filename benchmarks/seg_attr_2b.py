"""2B 尺度下的整模型 prefill 段归因 —— 为「TileLang 化做到什么范围」定范围。

为什么必须重做：A″ 的「残余 49.2% = 178 elementwise」是 4.7M/T=8192 的 profile。
FLOPs 账（已核，与 B/T 无关）显示 2B 的构成与 4.7M 完全不同：

  段                    4.7M(T=8192)   2B
  ① 块内注意              18.5%        10.4%
  ② 跨块 fold+检索        12.4%        33.3%
  ③ MLP 门控              18.5%        50.0%
  ④ 输出头                50.6%         6.3%

⚠️ FLOPs 账只说明「哪段值得看」，不说明「TileLang 能不能把那段做快」。
   ②③ 是 dense GEMM（cuBLAS 本来不差），TileLang 的赢面在**融合**（消中间物化/碎 kernel），
   这一点从未实测 → 本脚本取实测墙钟 + op 级中间物化。

口径：**eager 形态**（不为速度，为看构成与中间物化；compile 会改形态但不改构成）
      消融作为诊断，被消段换成「错但不崩」的廉价实现，只测时间不测正确性。
      每个变体同形态（都 eager），遵守「消融必同口径」。

用法: PYTHONPATH=/data/dynfw python benchmarks/seg_attr_2b.py [--T 1024] [--batch 1]
输出: /data/logs/seg_attr_2b.json + 控制台
"""
import sys
import json
import types
import argparse

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

torch._dynamo.config.cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM          # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5, FWAttentionOpt  # noqa: E402

# 2B 产品规格（MiniCPM5-2B 同族形态）；struct = 3*mm*D^2*nl = 2.114B
CFG2B = dict(D=2048, nh=16, n_layer=42, mlp_mult=4, vocab=130560, W=256)


def make_attn_fwd(no_intra=False, no_cross=False, no_rope=False):
    def fwd(self, Q, K, V, memories=None, W=512):
        assert K is Q
        B, nh, T, N = Q.size()
        D = self.D
        if W <= 0 or T % W != 0:
            raise NotImplementedError(f'要求 T % W == 0，实得 T={T} W={W}')
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

        if no_intra:                                   # ① 块内注意 → 廉价替身
            a = vv
        else:
            sc = torch.bmm(qq, qq.transpose(1, 2))
            causal = torch.tril(torch.ones(W, W, device=Q.device, dtype=torch.bool), 0)
            sc = sc.masked_fill(~causal, float('-inf'))
            p = torch.softmax(sc.float(), dim=-1)
            a = torch.bmm(p, vv)

        if no_cross:                                   # ②③④ 跨块 → 跳过
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
    def fwd(self, x, memories=None):
        C = self.config
        B = x.shape[0]; T = x.shape[2]
        D = self.D; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.ln(x)
        for _ in range(self.steps):
            x_sparse = F.relu(x @ self.encoder)
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x,
                                     memories=memories, W=self.W)
            yKV = self.ln(yKV)
            if no_mlp:
                yMLP = x.reshape(B, 1, T, D)
            else:
                y_sparse = F.relu(yKV @ self.encoder_v)
                xy = self.drop(x_sparse * y_sparse)
                yMLP = xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            x = self.ln(x + self.ln(yMLP))
            memories = new_mem
        return x, memories
    return fwd


def lm_fwd_nohead(self, x, targets=None):
    B, T = x.size()
    h = self.ln(self.e(x).unsqueeze(1))
    for blk in self.blocks:
        h, _ = blk(h, None)
    return h.view(B, T, self.D), None


def build(no_intra=False, no_cross=False, no_rope=False, no_mlp=False, no_head=False,
          cfg=None):
    c = dict(cfg or CFG2B)
    torch.manual_seed(0)
    m = BDHBlockFWCycleLM(D=c['D'], nh=c['nh'], vocab=c['vocab'],
                          n_layer=c['n_layer'], steps=1,
                          mlp_mult=c['mlp_mult'], W=c['W'])
    to_opt5(m, True, False, False)
    af = make_attn_fwd(no_intra, no_cross, no_rope)
    bf = make_block_fwd(no_mlp)
    for blk in m.blocks:
        blk.attn.forward = types.MethodType(af, blk.attn)
        blk.forward = types.MethodType(bf, blk)
    if no_head:
        m.forward = types.MethodType(lm_fwd_nohead, m)
    return m.cuda().eval()          # ⚠️ 不 compile：要的是 op 级可见性


def bench(m, T, B, rounds=3, iters=2, warmup=1):
    x = torch.randint(0, CFG2B['vocab'], (B, T), device='cuda')
    y = torch.randint(0, CFG2B['vocab'], (B, T), device='cuda')
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
            ev[1].record(); torch.cuda.synchronize()
            s.append(ev[0].elapsed_time(ev[1]) / iters)
    s.sort()
    return s[len(s) // 2], s[0]


def op_table(m, T, B, topn=22):
    x = torch.randint(0, CFG2B['vocab'], (B, T), device='cuda')
    y = torch.randint(0, CFG2B['vocab'], (B, T), device='cuda')
    with torch.no_grad():
        with autocast_ctx():
            m(x, y)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p:
            with autocast_ctx():
                m(x, y)
            torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.key, e.count,
             getattr(e, 'input_shapes', None)) for e in p.key_averages()
            if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    out = []
    for us, k, c, sh in rows[:topn]:
        s = ''
        if sh:
            try:
                s = str(sh[0])[:44]
            except Exception:
                s = '?'
        out.append((us / tot, us / 1000, c, k[:58], s))
    return out, tot / 1000, len(rows)


def autocast_ctx():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--T2', type=int, default=8192)
    a = ap.parse_args()
    T, B = a.T, a.batch

    print('=' * 100)
    print(f'2B 尺度段归因   D=2048 nh=16 nl=42 mm=4 N=512 vocab=130560')
    print(f'形态 = to_opt5 + **eager**（要 op 级可见性，不 compile）；前向 only，autocast bf16')
    print(f'B={B}  T={T}')
    print('=' * 100)

    res = {}
    variants = [('全量', dict()), ('-① 块内注意', dict(no_intra=True)),
                ('-② 跨块(fold+检索)', dict(no_cross=True)), ('-rope', dict(no_rope=True)),
                ('-③ MLP 门控', dict(no_mlp=True)), ('-④ 输出头', dict(no_head=True))]
    for name, kw in variants:
        try:
            m = build(**kw)
            med, mn = bench(m, T, B)
            res[name] = (med, mn)
            full = res.get('全量')
            d = f'Δ={full[0]-med:8.2f} ms  {((full[0]-med)/full[0]):6.1%}' if full else '(基准)'
            print(f'{name:<20}{med:10.2f} ms (min {mn:8.2f})   {d}')
            del m; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            res[name] = None
            print(f'{name:<20}{"OOM":>10}   {str(e)[:60]}')
            torch.cuda.empty_cache()

    json_out = {'cfg': CFG2B, 'T': T, 'B': B, 'form': 'to_opt5+eager', 'mode': 'forward_only'}
    full = res.get('全量')
    if full:
        print()
        print('-' * 100)
        print('【段占比（实测墙钟，eager 形态）】')
        json_out['shares'] = {}
        for k in ('-① 块内注意', '-② 跨块(fold+检索)', '-rope', '-③ MLP 门控', '-④ 输出头'):
            v = res.get(k)
            if v:
                sh = (full[0] - v[0]) / full[0]
                json_out['shares'][k] = sh
                print(f'   {k:<22} {sh:7.1%}   ({full[0]-v[0]:9.2f} ms)')
        print('-' * 100)

        # op 级：看中间物化（这正是 TileLang 融合能吃掉的东西）
        print('\n【op 级 top22（按 self CUDA 时间）—— 找中间物化与碎 kernel】')
        try:
            m = build()
            tbl, gpu_ms, nk = op_table(m, T, B)
            json_out['op_table'] = [{'pct': p, 'ms': ms, 'count': c, 'key': k, 'shape': s}
                                    for p, ms, c, k, s in tbl]
            json_out['gpu_self_total_ms'] = gpu_ms
            json_out['n_kernel_keys'] = nk
            print(f'   GPU self total {gpu_ms:.1f} ms，不同 kernel 种类 {nk} 个')
            for p, ms, c, k, s in tbl:
                print(f'   {p:6.1%} {ms:8.2f}ms {c:>7}  {k:<58} {s}')
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print('   op_table 失败:', type(e).__name__, str(e)[:120])

    # 长上下文对照（构成是否随 T 漂移）
    if a.T2 and a.T2 != T:
        print()
        print('=' * 100)
        print(f'【对照】T={a.T2} 下重测「全量 / -① / -②」（看构成是否随 T 漂移）')
        print('=' * 100)
        for name, kw in [('全量', dict()), ('-① 块内注意', dict(no_intra=True)),
                         ('-② 跨块', dict(no_cross=True))]:
            try:
                m = build(**kw)
                med, mn = bench(m, a.T2, B)
                res[f'{name}@T{a.T2}'] = (med, mn)
                print(f'{name:<20}@T{a.T2}{med:10.2f} ms')
                del m; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f'{name:<20}@T{a.T2}{"OOM":>10}')
                torch.cuda.empty_cache()

    json_out['results'] = {k: (None if v is None else {'median_ms': v[0], 'min_ms': v[1]})
                           for k, v in res.items()}
    with open('/data/logs/seg_attr_2b.json', 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print('\n已写入 /data/logs/seg_attr_2b.json')


if __name__ == '__main__':
    main()
