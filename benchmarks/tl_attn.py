"""TileLang 最小验证：把 v6 的【块内注意】用 TileLang 重写，与「我们 A″ 的交付」同口径对轰。

===================== 跑前锁死的判据（事后不改） =====================
对象（逐字对齐 dynfw/models/fused_fw_fw_cycle.py L60-73 的块内注意，一行数学不动）:
    sim  = q_c @ k_c.mT                    # q_c,k_c: [B,nh,w,N]，K 与 Q 同一份
    sim  = masked_fill(~tril(diag=0), -inf) # 因果，看自己及之前
    attn = softmax(sim.float(), dim=-1)     # fp32 softmax
    agg  = attn @ v_c                       # v_c: [B,1,w,D] 对 nh 广播
⚠️ 无 √N 缩放（本架构不做缩放）—— 这是最容易抄错的一点
⚠️ 范围只含「块内注意」；跨块 fast-weight 检索/更新不在本次范围

对照组（三方，同形状同 dtype 同 GPU）:
    A 纯 eager PyTorch（基线条，就是上面四行）
    B A + torch.compile            ← 我们 A″ 的实际交付形态
    C TileLang 手写 kernel（本次候选）

各配置（B=2, nh=8, w=256, N=128, D=256）:
  ⚠️ fp32 路径不可用：TileLang 的 fp32 T.gemm 在本结构下报 "Layout infer conflict
     between acc_s and acc_c"（实测 bm/bn ∈ {32,64,128} 四组全失败）→ 只能走 bf16。
     因此判据由「fp32 同路径 maxdiff」改为下面这套【结构性精确判据】：
    1a 首行恒等：位置 0 只看得见自己 ⇒ out[:,:,0,:] 必须逐位 == v[:,0,0,:]（判 r0 == 0）
    1b 因果扰动 ：只改 v[w//2]，位置 < w//2 的输出必须逐位不变（污染数须为 0）
    1c 误差地板 ：TileLang vs bf16-eager 的偏离 ≤ max(bf16 自身地板×3, 1e-3)
    1d top-1 一致率（部署口径，仅报数不作判据）
    性能: CUDA events 计时，5 轮 × 50 次，取中位；每个 block 组合都须先过 1a 才算数
=====================================================================

判定规则（先写死）:
    C 相对 B 加速比 ≥1.15× → TileLang 路线对我们的块内注意"值得继续"
    C 相对 B 在 0.87~1.15× → "打平"，属可选工具（降本价值在可维护性而非性能）
    C 相对 B <0.87×       → "不划算"，如实记录，不算失败
    闸门1 FAIL → 一律不报性能（数字无意义）
=====================================================================
用法: python benchmarks/tl_attn.py --batch 2 --w 256 [--no-tilelang]
"""
import argparse, math, sys, time, json, statistics
import torch, torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')

B, NH, W, N, D = 2, 8, 256, 128, 256
LOG2E = 1.4426950408889634


# ---------------- 参照：v6 块内注意（逐字对齐 L60-73） ----------------
def attn_ref(q, k, v):
    """q,k: [B,nh,w,N]  v: [B,1,w,D]；softmax 走 fp32；无 √d 缩放"""
    sim = q @ k.mT
    w_ = q.shape[2]
    causal = torch.tril(torch.ones(w_, w_, device=q.device, dtype=torch.bool), diagonal=0)
    sim = sim.masked_fill(~causal, float('-inf'))
    attn = torch.softmax(sim.float(), dim=-1)
    return attn @ v                      # fp32 @ bf16 -> fp32（与基线一致）


# ---------------- TileLang kernel ----------------
def build_tilelang(batch, heads, S, N_, D_, dt, block_M=64, block_N=64, threads=128, num_stages=1):
    import tilelang
    import tilelang.language as T

    dq = {'fp32': T.float32, 'bf16': T.bfloat16}[dt]
    accum = T.float32
    scale = LOG2E                        # exp2 换底：等价 exp(x)，且【不含】1/sqrt(N)

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
                m_cur = T.alloc_fragment([block_M], accum)
                m_prev = T.alloc_fragment([block_M], accum)
                r_sc = T.alloc_fragment([block_M], accum)
                l_sum = T.alloc_fragment([block_M], accum)
                lgsum = T.alloc_fragment([block_M], accum)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Qs)
                T.fill(acc_o, 0)
                T.fill(lgsum, 0)
                T.fill(m_cur, -T.infinity(accum))

                loop = T.min(T.ceildiv(S, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                for kk in T.Pipelined(loop, num_stages=num_stages):
                    T.copy(K[bz, by, kk * block_N:(kk + 1) * block_N, :], Ks)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= kk * block_N + j, 0, -T.infinity(accum))
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(m_cur, m_prev)
                    T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        m_cur[i] = T.max(m_cur[i], m_prev[i])
                    for i in T.Parallel(block_M):
                        r_sc[i] = T.exp2(m_prev[i] * scale - m_cur[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_cur[i] * scale)
                    T.reduce_sum(acc_s, l_sum, dim=1)
                    for i in T.Parallel(block_M):
                        lgsum[i] = lgsum[i] * r_sc[i] + l_sum[i]
                    T.copy(acc_s, acc_c)
                    for i, j in T.Parallel(block_M, D_):
                        acc_o[i, j] *= r_sc[i]
                    T.copy(V[bz, 0, kk * block_N:(kk + 1) * block_N, :], Vs)
                    T.gemm(acc_c, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, D_):
                    acc_o[i, j] /= lgsum[i]
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
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record(); torch.cuda.synchronize()
        meds.append(st.elapsed_time(en) / iters)
    return statistics.median(meds)


def kcount(fn, n=11):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=B)
    ap.add_argument('--w', type=int, default=W)
    ap.add_argument('--no-tilelang', action='store_true')
    ap.add_argument('--gate-only', action='store_true', help='只跑正确性闸门，不测性能（GPU 忙时用）')
    ap.add_argument('--bm', type=int, default=64)
    ap.add_argument('--bn', type=int, default=64)
    a = ap.parse_args()
    b, w_ = a.batch, a.w
    torch.manual_seed(0)
    dev = 'cuda'
    print('=' * 100)
    print(f'TileLang vs 现有实现 —— v6 块内注意  B={b} nh={NH} w={w_} N={N} D={D}')
    print(f'GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability()}')
    print('=' * 100)

    # ⚠️ 数据尺度必须让 logits ~ O(1)：q,k 取 N(0,1) 时 sim=q@k^T 的 std≈sqrt(N)=11.3
    #    → softmax 退化成近似 one-hot → 因果扰动探针只会检出 1 行（我第一版就踩了这个）。
    #    本架构【不做 1/sqrt(N) 缩放】，所以把尺度压进【数据】里（公式一行不动）。
    qf = torch.randn(b, NH, w_, N, device=dev, dtype=torch.float32) * (N ** -0.25)
    vf = torch.randn(b, 1, w_, D, device=dev, dtype=torch.float32)
    ref_f = attn_ref(qf, qf, vf)
    qb = qf.bfloat16(); vb = vf.bfloat16()
    kbf = None
    print(f'   数据尺度: q,k ~ N(0,1/N) → sim std≈{ (qf@qf.mT).std().item():.2f}（目标 O(1)）')

    # ================= 闸门1：结构正确性（bf16 也能判，且比 maxdiff 严格）=================
    # 背景：TileLang 的 fp32 T.gemm 布局推断在本结构下报 Layout infer conflict（实测 bm/bn 32/64/128 全失败），
    #       所以不能用「fp32 同路径 maxdiff」当闸门。改用两条【精确结构判据】+ 一条误差地板对照。
    print('\n[闸门1] 结构正确性（bf16 域，精确判据）')
    tl_ok = False
    if a.no_tilelang:
        print('   (--no-tilelang 跳过)')
    else:
        try:
            kbf = build_tilelang(b, NH, w_, N, D, 'bf16', a.bm, a.bn)
            qb = qf.bfloat16(); vb = vf.bfloat16()
            got = kbf(qb, qb, vb)

            # 1a 首行恒等：位置 0 只看得见自己 → softmax 权重恒为 1 → out[0] 必须逐位等于 v[0]
            r0 = (got[:, :, 0, :].float() - vb[:, 0, 0, :].unsqueeze(1).float()).abs().max().item()
            print(f'   1a 首行恒等 out[:, :, 0, :] vs v[:, 0, 0, :]   maxdiff={r0:.3e}  '
                  f'{"PASS" if r0 == 0.0 else "FAIL"}')

            # 1b 因果扰动探针：只改 v[j]（j=w//2），位置 < j 的输出必须【逐位不变】
            j = w_ // 2
            vb2 = vb.clone(); vb2[:, :, j, :] += 1.0
            got2 = kbf(qb, qb, vb2)
            d = (got.float() - got2.float()).abs().amax(dim=(0, 1, 3))   # [w]
            leak = int((d[:j] > 0).sum().item())
            hit = int((d[j:] > 0).sum().item())
            print(f'   1b 因果扰动 改 v[{j}]  →  污染位置(<{j})={leak} 个（须为0）, '
                  f'应受影响位(≥{j})={hit}/{w_-j}  '
                  f'{"PASS" if leak == 0 and hit > 0 else "FAIL"}')

            # 1c 误差地板：C 相对 bf16-eager 的偏离，不得显著大于 bf16-eager 自身相对 fp32 的偏离
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ref_bf = attn_ref(qb, qb, vb).float()
            e_floor = (ref_bf - ref_f).abs().max().item() / ref_f.abs().max().item()
            e_tl = (got.float() - ref_bf).abs().max().item() / ref_f.abs().max().item()
            print(f'   1c 误差地板  bf16-eager vs fp32 = {e_floor:.3e}；TileLang vs bf16-eager = {e_tl:.3e}  '
                  f'{"PASS" if e_tl <= max(e_floor * 3, 1e-3) else "FAIL"}')

            # 1d top-1 一致率（部署口径）
            t1 = (got.float().argmax(-1) == ref_bf.argmax(-1)).float().mean().item()
            print(f'   1d top-1 一致率 (vs bf16-eager)                 {t1:6.2%}')

            tl_ok = (r0 == 0.0) and (leak == 0) and (hit > 0) and (e_tl <= max(e_floor * 3, 1e-3))
            print(f'   → 闸门1 {"PASS" if tl_ok else "FAIL"}')
        except Exception as e:
            print(f'   C TileLang 构建/运行失败: {type(e).__name__}: {str(e).splitlines()[0][:250]}')
            tl_ok = False

    # 顺手探一下 fp32 路径到底哪个 block 组合能过（仅为记录，不作闸门）
    if not a.no_tilelang:
        ok32 = []
        for bm_, bn_ in [(64, 64), (32, 32), (128, 64), (64, 32)]:
            try:
                kk = build_tilelang(b, NH, w_, N, D, 'fp32', bm_, bn_)
                o = kk(qf, qf, vf)
                md = (o - ref_f).abs().max().item()
                ok32.append(f'{bm_}x{bn_}:OK({md:.2e})')
            except Exception:
                ok32.append(f'{bm_}x{bn_}:X')
        print(f'   [记录] fp32 路径 block 组合: {" ".join(ok32)}')

    # ---------- 性能（只有闸门1 PASS 才报） ----------
    if a.gate_only:
        print('\n--gate-only：跳过性能测量（GPU 忙时不许取墙钟）')
        return
    if not tl_ok:
        print('\n闸门1 未通过（或 TileLang 不可用）→ 按判据【不报性能】')
        return
    print('\n[性能] CUDA events，5 轮 × 50 次，取中位')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        fA = lambda: attn_ref(qb, qb, vb)
        mB = torch.compile(attn_ref, dynamic=False)
        mB(qb, qb, vb)      # 触发编译
        torch.cuda.synchronize()
        fB = lambda: mB(qb, qb, vb)
        tA, tB = timeit(fA), timeit(fB)
        # ⚠️ kcount 必须留在 autocast 块内 —— 否则 attn_ref 里 fp32@bf16 会抛
        #    "expected scalar type BFloat16 but found Float"（我在这一行上连踩三次）
        kA, kB = kcount(fA), kcount(fB)

        # TileLang 扫 block 组合（每个都要过 1a 首行恒等才算数）
        cfgs, rows = [(64, 64), (128, 64), (64, 128), (128, 128), (256, 64), (64, 256)], []
        v0 = vb[:, 0, 0, :].unsqueeze(1).float()
        for bm_, bn_ in cfgs:
            try:
                kk = build_tilelang(b, NH, w_, N, D, 'bf16', bm_, bn_)
                o = kk(qb, qb, vb)
                r0_ = (o[:, :, 0, :].float() - v0).abs().max().item()
                tt = timeit(lambda: kk(qb, qb, vb))
                rows.append((bm_, bn_, tt, r0_, kcount(lambda: kk(qb, qb, vb))))
            except Exception as e:
                rows.append((bm_, bn_, None, None, str(e).splitlines()[0][:60]))
    print(f'   A eager        {tA:8.4f} ms   {kA:6.1f} kernel/call')
    print(f'   B compile      {tB:8.4f} ms   {kB:6.1f} kernel/call')
    print(f'   {"C TileLang (block_M x block_N)":<30}{"ms":>10}{"kernel":>9}   首行闸门')
    best = None
    for bm_, bn_, tt, r0_, kk_ in rows:
        if tt is None:
            print(f'   {f"C {bm_}x{bn_}":<30}{"编译/运行失败":>10}   {kk_}')
        else:
            ok = 'PASS' if r0_ == 0.0 else f'FAIL({r0_:.1e})'
            print(f'   {f"C {bm_}x{bn_}":<30}{tt:>10.4f}{kk_:>9.1f}   {ok}')
            if r0_ == 0.0 and (best is None or tt < best[2]):
                best = (bm_, bn_, tt)
    if best is None:
        print('\n   没有任何 TileLang 配置通过首行闸门 → 按判据不报加速比')
        return
    bm_, bn_, tC = best
    r = tB / tC
    verdict = ('值得继续' if r >= 1.15 else ('打平/可选工具' if r >= 0.87 else '不划算'))
    print(f'\n   最佳 C = {bm_}x{bn_}  {tC:.4f} ms')
    print(f'   → C/B = {r:.3f}x   (vs eager {tA/tC:.3f}x)   判定: {verdict}')
    json.dump(dict(batch=b, w=w_, N=N, D=D, best_block=[bm_, bn_],
                   tA=tA, tB=tB, tC=tC, kA=kA, kB=kB, ratio_C_over_B=r, verdict=verdict,
                   all=[{'bm': x[0], 'bn': x[1], 'ms': x[2], 'row0': x[3]} for x in rows if x[2]]),
              open('/data/logs/tl_attn.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
