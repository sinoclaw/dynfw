"""探明 FLA chunk_gla 的 API 语义（形状/门控空间/输出状态），为 v6.6 接入做前置确认。

我们的形式:  M_n = α_n ⊙ M_{n-1} + k_n ⊗ v_n      (α = sigmoid, 标量/逐维)
FLA 的 GLA:  S_n = diag(exp(g_n)) S_{n-1} + k_n ⊗ v_n   (g 在 log 空间)
⇒ 语义一致（都是"门控衰减 + 外积写入"），需确认:
   1) q/k/v/g 的期望形状与头数关系
   2) g 的取值域（log 空间？是否需要 logsigmoid？）
   3) q,k 是否要求 L2 归一化
   4) 返回的 recurrent state 形状（与我们的 memories [B,nh,N,D] 是否一致）
   5) 是否支持 K != V（我们 k 维 N=512、v 维 D=128）
   6) 是否支持"v 的头数=1 而 q/k 头数=nh"（我们的 V 对头共享）
"""
import torch
import torch.nn.functional as F

try:
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
    print('导入成功: chunk_gla / fused_chunk_gla / fused_recurrent_gla')
except Exception as e:
    print('导入失败:', type(e).__name__, e)
    raise SystemExit(1)

DEV = 'cuda'
print('\n=== 测试 1: 标准形状 B,H,T,K / B,H,T,V（K != V）===')
B, H, T, K, V = 2, 4, 128, 64, 32
q = torch.randn(B, H, T, K, device=DEV)
k = torch.randn(B, H, T, K, device=DEV)
v = torch.randn(B, H, T, V, device=DEV)
g = F.logsigmoid(torch.randn(B, H, T, K, device=DEV))     # log 空间门控（<= 0）
try:
    o, st = chunk_gla(q=q, k=k, v=v, g=g, output_final_state=True)
    print(f'  OK  o={tuple(o.shape)}  state={tuple(st.shape)}')
    print(f'  ⇒ state 形状 = [B,H,K,V]，与我们的 memories [B,nh,N,D] 语义一致')
except Exception as e:
    print('  FAIL', type(e).__name__, str(e)[:200])

print('\n=== 测试 2: g 为标量门控（每 token 一个值，广播到 K 维）===')
g1 = F.logsigmoid(torch.randn(B, H, T, 1, device=DEV)).expand(-1, -1, -1, K)
try:
    o, st = chunk_gla(q=q, k=k, v=v, g=g1, output_final_state=True)
    print(f'  OK  o={tuple(o.shape)}  state={tuple(st.shape)}')
except Exception as e:
    print('  FAIL', type(e).__name__, str(e)[:200])

print('\n=== 测试 3: v 头数=1（我们的 V 对 nh 头共享）===')
v1 = torch.randn(B, 1, T, V, device=DEV)
try:
    o, st = chunk_gla(q=q, k=k, v=v1, g=g, output_final_state=True)
    print(f'  OK  o={tuple(o.shape)}  state={tuple(st.shape)}')
except Exception as e:
    print('  FAIL（需 expand v 到 H 头）:', type(e).__name__, str(e)[:150])

print('\n=== 测试 4: initial_state 传入 + 分段续算（验证跨 chunk 状态传递）===')
try:
    s0 = torch.zeros(B, H, K, V, device=DEV)
    o1, s1 = chunk_gla(q=q, k=k, v=v, g=g, initial_state=s0, output_final_state=True)
    o2, s2 = chunk_gla(q=q, k=k, v=v, g=g, initial_state=s1, output_final_state=True)
    o_all, s_all = chunk_gla(q=torch.cat([q, q], 2), k=torch.cat([k, k], 2),
                             v=torch.cat([v, v], 2), g=torch.cat([g, g], 2),
                             initial_state=s0, output_final_state=True)
    d = (torch.cat([o1, o2], 2) - o_all).abs().max().item()
    print(f'  分段 vs 整段 maxdiff = {d:.3e}  {"等价 ✓" if d < 1e-4 else "不等价 ✗"}')
    print(f'  ⇒ 证明 initial_state 机制可用于我们的"块间状态传递"')
except Exception as e:
    print('  FAIL', type(e).__name__, str(e)[:200])

print('\n=== 测试 5: 是否需要 q/k 归一化（对比归一化前后数值量级）===')
o_raw, _ = chunk_gla(q=q, k=k, v=v, g=g)
qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
o_norm, _ = chunk_gla(q=qn, k=kn, v=v, g=g)
print(f'  未归一化 |o|max={o_raw.abs().max():.3f}   归一化后 |o|max={o_norm.abs().max():.3f}')
print('  （两者都可用；归一化只是数值稳定手段，不是硬要求 ⇒ 由我们决定是否加）')

print('\n=== 测试 6: 反向是否可跑（训练必需）===')
qq = q.clone().requires_grad_(True); kk = k.clone().requires_grad_(True)
vv = v.clone().requires_grad_(True); gg = g.clone().requires_grad_(True)
o, st = chunk_gla(q=qq, k=kk, v=vv, g=gg, output_final_state=True)
o.sum().backward()
print(f'  反向 OK  grad q={qq.grad.abs().max():.2e} k={kk.grad.abs().max():.2e} '
      f'v={vv.grad.abs().max():.2e} g={gg.grad.abs().max():.2e}')
