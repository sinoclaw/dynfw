"""FLA chunk_gla 语义确认（第二版）。

第一版的问题:
  1) 未传 head_first=True → FLA 把 [B,H,T,K] 误读为 [B,T,H,...]（形状警告已提示）
  2) 测试 4 崩了之后 CUDA 上下文损坏，后续测试全废
本版:
  1) 显式 head_first=True
  2) 每个测试独立进程（由外层 shell 调用，--case N），避免崩溃互相污染
  3) **朴素逐 token 实现做数值对照** —— 不只验"能跑"，更验"算的是不是我们要的东西"

语义目标（我们的 v6.6 形式）:
  S_t = diag(exp(g_t)) · S_{t-1} + k_t ⊗ v_t
  o_t = q_t @ S_t           ← 注意：是"含当前 token 写完之后"还是"写之前"，用对照实验定
"""
import argparse
import sys

import torch
import torch.nn.functional as F

DEV = 'cuda'


def naive_gla(q, k, v, g, o_after_write=True):
    """朴素逐 token 参考实现（fp32），用于确认 FLA 的语义/时序。"""
    B, H, T, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    outs = []
    q_, k_, v_, g_ = q.float(), k.float(), v.float(), g.float()
    for t in range(T):
        if not o_after_write:
            outs.append(torch.einsum('bhk,bhkv->bhv', q_[:, :, t], S))
        decay = torch.exp(g_[:, :, t]).unsqueeze(-1)                 # [B,H,K,1]
        S = decay * S + k_[:, :, t].unsqueeze(-1) * v_[:, :, t].unsqueeze(-2)
        if o_after_write:
            outs.append(torch.einsum('bhk,bhkv->bhv', q_[:, :, t], S))
    return torch.stack(outs, 2), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, default=0)
    a = ap.parse_args()

    from fla.ops.gla import chunk_gla

    torch.manual_seed(0)
    B, H, T, K, V = 2, 4, 128, 64, 32
    q = torch.randn(B, H, T, K, device=DEV)
    k = torch.randn(B, H, T, K, device=DEV)
    v = torch.randn(B, H, T, V, device=DEV)
    g = F.logsigmoid(torch.randn(B, H, T, K, device=DEV))      # log 空间（<=0，衰减）

    if a.case == 0:
        print('[case 0] head_first=True 基本可用性')
        o, st = chunk_gla(q=q, k=k, v=v, g=g, head_first=True, output_final_state=True)
        print(f'  o={tuple(o.shape)} state={tuple(st.shape)}')

    elif a.case == 1:
        print('[case 1] 数值语义对照（FLA vs 朴素）')
        o, st = chunk_gla(q=q, k=k, v=v, g=g, head_first=True, output_final_state=True)
        for after in (True, False):
            on, sn = naive_gla(q, k, v, g, o_after_write=after)
            do = (o.float() - on).abs().max().item()
            ds = (st.float() - sn).abs().max().item()
            tag = 'o_t = q_t @ S_t（写之后）' if after else 'o_t = q_t @ S_{t-1}（写之前）'
            print(f'  假设 {tag:28s} maxdiff(o)={do:.3e}  maxdiff(state)={ds:.3e}')
        print('  ⇒ 谁接近 0 就是 FLA 的真实语义（这决定我们怎么对齐 v6.6 的读侧时序）')

    elif a.case == 2:
        print('[case 2] initial_state 跨段续算等价性')
        s0 = torch.zeros(B, H, K, V, device=DEV)
        o1, s1 = chunk_gla(q=q, k=k, v=v, g=g, initial_state=s0, head_first=True, output_final_state=True)
        o2, s2 = chunk_gla(q=q, k=k, v=v, g=g, initial_state=s1, head_first=True, output_final_state=True)
        o_all, s_all = chunk_gla(q=torch.cat([q, q], 2), k=torch.cat([k, k], 2),
                                 v=torch.cat([v, v], 2), g=torch.cat([g, g], 2),
                                 initial_state=s0, head_first=True, output_final_state=True)
        d1 = (torch.cat([o1, o2], 2) - o_all).abs().max().item()
        d2 = (s2 - s_all).abs().max().item()
        print(f'  maxdiff(o)={d1:.3e}  maxdiff(state)={d2:.3e}  '
              f'{"等价 ✓（可做块间状态传递）" if max(d1, d2) < 1e-3 else "不等价 ✗"}')

    elif a.case == 3:
        print('[case 3] v 头数=1（我们的 V 对 nh 头共享）')
        v1 = torch.randn(B, 1, T, V, device=DEV)
        o, st = chunk_gla(q=q, k=k, v=v1, g=g, head_first=True, output_final_state=True)
        print(f'  o={tuple(o.shape)} state={tuple(st.shape)}')
        o2, st2 = chunk_gla(q=q, k=k, v=v1.expand(-1, H, -1, -1), g=g, head_first=True, output_final_state=True)
        d = (st - st2).abs().max().item()
        print(f'  vs v 显式 expand 到 H 头: maxdiff(state)={d:.3e}')

    elif a.case == 4:
        print('[case 4] 反向可跑（训练必需）')
        qq = q.clone().requires_grad_(True); kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True); gg = g.clone().requires_grad_(True)
        o, st = chunk_gla(q=qq, k=kk, v=vv, g=gg, head_first=True, output_final_state=True)
        o.sum().backward()
        print(f'  OK  grad|q|={qq.grad.abs().max():.2e} |k|={kk.grad.abs().max():.2e} '
              f'|v|={vv.grad.abs().max():.2e} |g|={gg.grad.abs().max():.2e}')

    elif a.case == 5:
        print('[case 5] 长序列 + 大 K/V（贴近我们：T=8192, K=N=512, V=D=128, H=16）')
        import time
        B2, H2, T2, K2, V2 = 1, 16, 8192, 512, 128
        q2 = torch.randn(B2, H2, T2, K2, device=DEV)
        k2 = torch.randn(B2, H2, T2, K2, device=DEV)
        v2 = torch.randn(B2, H2, T2, V2, device=DEV)
        g2 = F.logsigmoid(torch.randn(B2, H2, T2, K2, device=DEV))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o2, st2 = chunk_gla(q=q2, k=k2, v=v2, g=g2, head_first=True, output_final_state=True)
        torch.cuda.synchronize()
        print(f'  前向 OK  o={tuple(o2.shape)} state={tuple(st2.shape)}  '
              f'耗时 {(time.perf_counter()-t0)*1000:.1f}ms  peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
        loss = o2.float().sum()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        print(f'  反向 OK  耗时 {(time.perf_counter()-t0)*1000:.1f}ms')


if __name__ == '__main__':
    main()
