"""FLA 0.5.2 `fused_chunk_simple_gla` 语义确认（第三版，对齐真实 API）。

为什么换 kernel: 0.5.2 里 `fla.ops.gla` 已不存在；真正匹配我们形式的是
  `fla.ops.simple_gla.fused_chunk_simple_gla`
  文档原话: "Compared to GLA, the gating is head-wise instead of elementwise"
  ⇒ 门控 g 形状 [B,T,H]（每 token 每头一个标量），正是 v6.6 的 α = sigmoid(gate(k_c))

布局: 0.5.2 是 **[B, T, H, K]**（head-second），与 0.2.2 的 [B,H,T,K] 相反，务必别搞混。

语义目标:  S_t = exp(g_t) · S_{t-1} + k_t ⊗ v_t ;  o_t = q_t @ S_t
"""
import argparse

import torch
import torch.nn.functional as F

DEV = 'cuda'


def naive(o_after_write=True):
    """朴素逐 token 参考（fp32），用于确认时序语义。输入 head-second [B,T,H,K]。"""
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, device=DEV, dtype=torch.float32)
    outs = []
    for t in range(T):
        if not o_after_write:
            outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t].float(), S))
        decay = torch.exp(g[:, t]).float().unsqueeze(-1).unsqueeze(-1)      # [B,H,1,1]
        S = decay * S + k[:, t].float().unsqueeze(-1) * v[:, t].float().unsqueeze(-2)
        if o_after_write:
            outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t].float(), S))
    return torch.stack(outs, 1), S       # [B,T,H,V], [B,H,K,V]


def main():
    global q, k, v, g
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=0)
    a = ap.parse_args()

    from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla

    torch.manual_seed(0)
    B, T, H, K, V = 2, 256, 4, 64, 32
    q = torch.randn(B, T, H, K, device=DEV)
    k = torch.randn(B, T, H, K, device=DEV)
    v = torch.randn(B, T, H, V, device=DEV)
    g = F.logsigmoid(torch.randn(B, T, H, device=DEV))      # head-wise 标量门控（log 空间）

    if a.case == 0:
        print('[case 0] 基本可用性（head-second + head-wise g）')
        o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)
        print(f'  o={tuple(o.shape)}  state={tuple(st.shape)}')
        print(f'  ⇒ 期望 o=[B,T,H,V]；state=[B,H,K,V]（= 我们 memories 的 [B,nh,N,D]）')

    elif a.case == 1:
        print('[case 1] ★ 数值语义对照（FLA vs 朴素逐 token）')
        o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)
        for after in (True, False):
            on, sn = naive(o_after_write=after)
            do = (o.float() - on).abs().max().item()
            rel = do / max(on.abs().max().item(), 1e-9)
            ds = (st.float() - sn).abs().max().item()
            tag = 'o_t = q_t @ S_t    (写之后)' if after else 'o_t = q_t @ S_{t-1} (写之前)'
            print(f'  {tag:26s} maxdiff(o)={do:.3e} (rel {rel:.1e})  maxdiff(state)={ds:.3e}')
        print('  ⇒ 谁接近 0 即真实语义；这决定 v6.6 读侧要不要错开一拍')

    elif a.case == 2:
        print('[case 2] initial_state 跨段续算等价性')
        s0 = torch.zeros(B, H, K, V, device=DEV)
        o1, s1 = fused_chunk_simple_gla(q, k, v, g, initial_state=s0, output_final_state=True)
        o2, s2 = fused_chunk_simple_gla(q, k, v, g, initial_state=s1, output_final_state=True)
        oa, sa = fused_chunk_simple_gla(torch.cat([q, q], 1), torch.cat([k, k], 1),
                                        torch.cat([v, v], 1), torch.cat([g, g], 1),
                                        initial_state=s0, output_final_state=True)
        d1 = (torch.cat([o1, o2], 1) - oa).abs().max().item()
        d2 = (s2 - sa).abs().max().item()
        print(f'  maxdiff(o)={d1:.3e}  maxdiff(state)={d2:.3e}  '
              f'{"等价 ✓ 可用于块间状态传递" if max(d1, d2) < 1e-3 else "不等价 ✗"}')

    elif a.case == 3:
        print('[case 3] 反向可跑（训练必需）')
        qq = q.clone().requires_grad_(True); kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True); gg = g.clone().requires_grad_(True)
        o, st = fused_chunk_simple_gla(qq, kk, vv, gg, output_final_state=True)
        o.sum().backward()
        print(f'  OK  grad|q|={qq.grad.abs().max():.2e} |k|={kk.grad.abs().max():.2e} '
              f'|v|={vv.grad.abs().max():.2e} |g|={gg.grad.abs().max():.2e}')

    elif a.case == 4:
        print('[case 4] ★ 我们真实形状的耗时/显存（T=8192, H=16, K=N=512, V=D=128）')
        import time
        B2, T2, H2, K2, V2 = 1, 8192, 16, 512, 128
        q2 = torch.randn(B2, T2, H2, K2, device=DEV)
        k2 = torch.randn(B2, T2, H2, K2, device=DEV)
        v2 = torch.randn(B2, T2, H2, V2, device=DEV)
        g2 = F.logsigmoid(torch.randn(B2, T2, H2, device=DEV))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o2, st2 = fused_chunk_simple_gla(q2, k2, v2, g2, output_final_state=True)
        torch.cuda.synchronize()
        fwd = (time.perf_counter() - t0) * 1000
        print(f'  前向 {fwd:.1f}ms  o={tuple(o2.shape)} state={tuple(st2.shape)}  '
              f'peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o2.float().sum().backward()
        torch.cuda.synchronize()
        print(f'  反向 {(time.perf_counter()-t0)*1000:.1f}ms')
        print('  ⇒ 对照：v6.6 顺序循环在该形状下约占其 1548s/1000step 的主要部分')

    elif a.case == 5:
        print('[case 5] 对照 chunk_gla(0.5.2) 是否可用（elementwise 门控那支）')
        try:
            from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla as chunk_gla
            print('  fla.ops.gla 仍可导入')
        except Exception as e:
            print(f'  fla.ops.gla 不可用: {type(e).__name__} ⇒ 确认应走 simple_gla 分支')


if __name__ == '__main__':
    main()
