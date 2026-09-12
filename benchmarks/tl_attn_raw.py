"""B 线：把 v6 的【块内注意 —— **raw 语义**】用 TileLang 重写，与 A 线交付同口径对轰。

===================== 跑前锁死的判据（事后不改） =====================
对象（逐字对齐 dynfw/models/fused_fw_fw_cycle.py 的 read_mode='raw' 分支）:
    sim  = q_c @ k_c.mT                          # q,k: [B,nh,w,N]，K 与 Q 同一份
    sim  = masked_fill(~tril(diag=-1), 0.0)      # 因果：位置 i 只看 j<i（**不看自己**），掩码值是真 0
    agg  = sim @ v_c                             # v: [B,1,w,D] 对 nh 广播；**无 softmax、无归一化**
⚠️ 无 √N 缩放、无 softmax、无 online max/l 归约 —— 这正是 raw 比 flash 省的地方
⚠️ 范围只含「块内注意」；跨块 fast-weight 检索/更新不在本次范围

对照组（三方，同形状同 dtype 同 GPU）:
    A 纯 eager PyTorch（基线条，就是上面三行）
    B A + torch.compile            ← A 线（方案A）的实际交付形态
    C TileLang 手写 kernel（本次候选）

配置: B=2, nh=8, w=256, N=128, D=256（与 tl_attn.py 同）

【结构性精确判据（bf16 域也能判，比 maxdiff 严格）】
    1a 首行恒等（raw 特有）：diagonal=-1 ⇒ 位置 0 看不见任何 token（连自己都不看）
                             ⇒ out[:,:,0,:] 必须**逐位 == 0**
    1b 因果扰动：只改 v[j]（j=w//2），位置 **≤ j** 的输出必须逐位不变（位置 j 不看自己）
                 污染数须为 0；位置 > j 必须受影响（>0）
    1c 误差地板：TileLang vs bf16-eager 的偏离 ≤ max(bf16 自身地板×3, 1e-3)
    1d top-1 一致率（部署口径，仅报数不作判据）
性能: CUDA events，5 轮 × 50 次取中位；须先过 1a 才报性能

判定规则（先写死）:
    C 相对 B 加速比 ≥1.15× → TileLang 路线值得继续
    0.87 ~ 1.15×           → 打平（价值在可维护性而非性能）
    < 0.87×                → 不划算，如实记录
=====================================================================
用法: PYTHONPATH=/data/dynfw python benchmarks/tl_attn_raw.py --batch 2 --w 256 [--no-tilelang]
"""
import argparse
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')

B, NH, W, N, D = 2, 8, 256, 128, 256


# ---------------- 参照：v6 块内注意 raw 语义 ----------------
def attn_ref_raw(q, k, v):
    """q,k: [B,nh,w,N]  v: [B,1,w,D]；raw：diag=-1、mask=0、无 softmax、无缩放"""
    sim = q @ k.mT
    w_ = q.shape[2]
    causal = torch.tril(torch.ones(w_, w_, device=q.device, dtype=torch.bool), diagonal=-1)
    sim = sim.masked_fill(~causal, 0.0)
    return sim @ v


def attn_ref_raw_expanded(q, k, v):
    """A 线交付形态：v 只物化展开一次（[B,1,w,D]→[B*nh,w,D]），再走 bmm。对齐 FWAttentionOpt5Raw。"""
    B, nh, w_, N = q.shape
    D_ = v.shape[-1]
    qq = q.reshape(B * nh, w_, N)
    vv = v.expand(B, nh, w_, D_).reshape(B * nh, w_, D_)
    sim = torch.bmm(qq, qq.transpose(1, 2))
    mask = torch.tril(torch.ones(w_, w_, device=q.device, dtype=torch.bool), diagonal=-1)
    sim = sim.masked_fill(~mask, 0.0)
    return torch.bmm(sim, vv).view(B, nh, w_, D_)


# ---------------- TileLang raw kernel ----------------
def build_tilelang_raw(batch, heads, S, N_, D_, dt, block_M=64, block_N=64, threads=128, num_stages=1):
    import tilelang
    import tilelang.language as T

    dq = {'fp32': T.float32, 'bf16': T.bfloat16}[dt]
    accum = T.float32

    @tilelang.jit(out_idx=[3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False})
    def kern(batch, heads, S, N_, D_, block_M, block_N, threads, num_stages):
        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, S, N_], dq),
            K: T.Tensor([batch, heads, S, N_], dq),
            V: T.Tensor([batch, 1, S, D_], dq),
            Out: T.Tensor([batch, heads, S, D_], dq),
        ):
            with T.Kernel(T.ceildiv(S, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Qs = T.alloc_shared([block_M, N_], dq)
                Ks = T.alloc_shared([block_N, N_], dq)
                Vs = T.alloc_shared([block_N, D_], dq)
                Os = T.alloc_shared([block_M, D_], dq)
                acc_s = T.alloc_fragment([block_M, block_N], accum)
                acc_c = T.alloc_fragment([block_M, block_N], dq)
                acc_o = T.alloc_fragment([block_M, D_], accum)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Qs)
                T.fill(acc_o, 0)

                # 只看 j < i：块 kk 的上界由 bx*block_M 决定（对角线所在块也要算）
                loop = T.min(T.ceildiv(S, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                for kk in T.Pipelined(loop, num_stages=num_stages):
                    T.copy(K[bz, by, kk * block_N:(kk + 1) * block_N, :], Ks)
                    T.fill(acc_s, 0)
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    # raw 因果掩码：**diagonal=-1**（不看自己），掩码值 = 真 0（非 -inf）
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i > kk * block_N + j, acc_s[i, j], 0.0)
                    T.copy(acc_s, acc_c)
                    T.copy(V[bz, 0, kk * block_N:(kk + 1) * block_N, :], Vs)
                    T.gemm(acc_c, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)

                T.copy(acc_o, Os)
                T.copy(Os, Out[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return main

    return kern(batch, heads, S, N_, D_, block_M, block_N, threads, num_stages)


def timeit(fn, reps=5, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(reps):
        st = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        meds.append(st.elapsed_time(en) / iters)
    return statistics.median(meds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=B)
    ap.add_argument('--w', type=int, default=W)
    ap.add_argument('--no-tilelang', action='store_true')
    ap.add_argument('--gate-only', action='store_true', help='只跑正确性闸门，不测性能')
    ap.add_argument('--bm', type=int, default=64)
    ap.add_argument('--bn', type=int, default=64)
    a = ap.parse_args()
    b, w_ = a.batch, a.w
    torch.manual_seed(0)
    dev = 'cuda'
    print('=' * 100)
    print(f'TileLang(raw) vs A线交付 —— v6 块内注意【raw 语义】  B={b} nh={NH} w={w_} N={N} D={D}')
    print(f'GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability()}')
    print('=' * 100)

    # raw 无 softmax ⇒ 不需要压低尺度（没有 one-hot 退化问题）；但仍保持与 tl_attn.py 同样的数据尺度以便对照
    qf = torch.randn(b, NH, w_, N, device=dev, dtype=torch.float32) * (N ** -0.25)
    vf = torch.randn(b, 1, w_, D, device=dev, dtype=torch.float32)
    ref_f = attn_ref_raw(qf, qf, vf)
    qb = qf.bfloat16()
    vb = vf.bfloat16()
    print(f'   sim std≈{(qf @ qf.mT).std().item():.2f}（raw 无 softmax，尺度不敏感）')

    print('\n[闸门1] 结构正确性（raw 语义，bf16 域精确判据）')
    if a.no_tilelang:
        print('   (--no-tilelang 跳过)')
    else:
        try:
            krn = build_tilelang_raw(b, NH, w_, N, D, 'bf16', a.bm, a.bn)
            got = krn(qb, qb, vb)

            # 1a 首行恒等（raw 特有）：位置 0 看不见任何 token → 输出必须全 0
            r0 = got[:, :, 0, :].float().abs().max().item()
            print(f'   1a 首行恒等 out[:, :, 0, :] 必须全 0   maxabs={r0:.3e}  '
                  f'{"PASS" if r0 == 0.0 else "FAIL"}')

            # 1b 因果扰动：改 v[j] → 位置 ≤ j 必须逐位不变（位置 j 不看自己）
            j = w_ // 2
            vb2 = vb.clone()
            vb2[:, :, j, :] += 1.0
            got2 = krn(qb, qb, vb2)
            d = (got.float() - got2.float()).abs().amax(dim=(0, 1, 3))
            leak = int((d[:j + 1] > 0).sum().item())    # 含 j 本身
            hit = int((d[j + 1:] > 0).sum().item())
            print(f'   1b 因果扰动 改 v[{j}]  →  污染位置(≤{j})={leak} 个（须为0）, '
                  f'应受影响位(>{j})={hit}/{w_ - j - 1}  '
                  f'{"PASS" if leak == 0 and hit > 0 else "FAIL"}')

            # 1c 误差地板
            ref_bf = attn_ref_raw(qb, qb, vb).float()
            e_floor = (ref_bf - ref_f).abs().max().item() / max(ref_f.abs().max().item(), 1e-9)
            e_tl = (got.float() - ref_bf).abs().max().item() / max(ref_f.abs().max().item(), 1e-9)
            print(f'   1c 误差地板  bf16-eager vs fp32 = {e_floor:.3e}；TileLang vs bf16-eager = {e_tl:.3e}  '
                  f'{"PASS" if e_tl <= max(e_floor * 3, 1e-3) else "FAIL"}')

            # 1d top-1 一致率（报数）
            t1 = (got.float().argmax(-1) == ref_bf.argmax(-1)).float().mean().item()
            print(f'   1d top-1 一致率 (仅报数) = {t1 * 100:.2f}%')
        except Exception as e:
            print(f'   TileLang 失败: {type(e).__name__}: {str(e)[:300]}')
            return

    if a.gate_only:
        return

    print('\n[闸门2] 性能（CUDA events，5×50 取中位）')
    A = lambda: attn_ref_raw(qb, qb, vb)                                   # noqa: E731
    # A' = A 线【真实交付形态】：先物化展开 v 到 nh，再 bmm（对齐 FWAttentionOpt5Raw）
    A2 = lambda: attn_ref_raw_expanded(qb, qb, vb)                         # noqa: E731
    Bc = torch.compile(lambda: attn_ref_raw_expanded(qb, qb, vb))          # 主对照：A线交付 + compile
    Bc_naive = torch.compile(lambda: attn_ref_raw(qb, qb, vb))            # 参考：naive 广播版 + compile
    tA = timeit(A)
    tA2 = timeit(A2)
    tBn = timeit(Bc_naive)
    tB = timeit(Bc)
    tC = timeit(lambda: krn(qb, qb, vb))
    print(f'   A  eager naive(广播 v)        {tA  * 1000:9.3f} ms')
    print(f'   A2 eager 物化展开 v           {tA2 * 1000:9.3f} ms   ← A线交付形态（未 compile）')
    print(f'   Bn compile(naive 广播 v)      {tBn * 1000:9.3f} ms')
    print(f'   B  compile(物化展开 v)        {tB  * 1000:9.3f} ms   ← **主对照：A 线实际交付**')
    print(f'   C  TileLang(raw kernel)       {tC  * 1000:9.3f} ms')
    sp = tB / tC
    print(f'\n   C 相对 B（A线交付+compile）加速比 = {sp:.3f}x')
    verdict = ('✅ 值得继续' if sp >= 1.15 else
               ('⚠️ 打平（价值在可维护性）' if sp >= 0.87 else '❌ 不划算，如实记录'))
    print(f'   判定: {verdict}')
    print(f'   （参考对照：C 相对 A2(eager 物化) = {tA2 / tC:.3f}x；'
          f'C 相对 A(naive 广播) = {tA / tC:.3f}x）')
    print('\n   ⚠️ 口径说明：naive 版每 head 广播展开 v（8× 冗余访存），A 线交付形态物化展开一次；')
    print('      与 TileLang 比必须用后者（同工作量），否则高估 kernel 收益。')


if __name__ == '__main__':
    main()
