"""profile v6.7 的注意力层：拆出各段耗时，定位"1.73× 而非 12×"的缺口。

隔离 bench 里跨块状态段加速 12.2×，端到端只有 1.73× ⇒ 瓶颈在别处。
候选：
  A. permute/contiguous 搬运（[B,nh,T,N] ↔ [B,T,nh,N]，且 v 要 expand 到 nh 头）
  B. 块内 raw 注意（sim = q_c @ k_c.mT; masked_fill; attn @ v_c）—— 我们没换掉它
  C. self 项计算与减法
  D. 模型其余部分（encoder/decoder/LN/head）

方法：用 torch.cuda.Event 对每个子段计时，同形状（T=8192, nh=16, N=512, D=128）。
"""
import time

import torch
import torch.nn.functional as F

DEV = 'cuda'
torch.manual_seed(0)

from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla

# 与正式实验一致：T=8192, W=64, D=128, nh=16, mlp_mult=64 ⇒ N = 64*128/16 = 512
T, W, B, nh, D, N = 8192, 64, 1, 16, 128, 512
Q = torch.randn(B, nh, T, N, device=DEV, dtype=torch.bfloat16)
V = torch.randn(B, 1, T, D, device=DEV, dtype=torch.bfloat16)
mem = torch.zeros(B, nh, N, D, device=DEV, dtype=torch.float32)
gate_w = torch.randn(N, 1, device=DEV) * 0.02
gate_b = torch.full((1,), 4.0, device=DEV)


def t_it(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


print(f'=== v6.7 注意力层分段 profile（T={T}, W={W}, nh={nh}, N={N}, D={D}）===')
print()

# ---- A. permute/搬运 ----
q_f = Q.permute(0, 2, 1, 3).contiguous()
v_exp = V.expand(-1, nh, -1, -1)
r = torch.arange(0, T, device=DEV, dtype=torch.float32).view(1, 1, -1, 1)


def a_move():
    a = Q.permute(0, 2, 1, 3).contiguous()
    b = V.expand(-1, nh, -1, -1).permute(0, 2, 1, 3).contiguous()
    return a, b


tA = t_it(a_move)

# ---- B. 块内 raw 注意（我们没换掉的部分，逐块循环）----
def b_blocks():
    outs = []
    for st in range(0, T, W):
        en = min(st + W, T)
        q_c = Q[:, :, st:en]; k_c = Q[:, :, st:en]; v_c = V[:, :, st:en]
        w = en - st
        sim = q_c @ k_c.mT
        causal = torch.tril(torch.ones(w, w, device=DEV, dtype=torch.bool), diagonal=-1)
        attn = sim.masked_fill(~causal, 0.0)
        outs.append(attn @ v_c)
    return torch.cat(outs, dim=2)


tB = t_it(b_blocks, warmup=2, reps=5)

# ---- C. FLA kernel 本体（含搬运后的输入）----
def c_fla():
    g = F.logsigmoid(torch.zeros(B, T, nh, device=DEV, dtype=torch.float32))
    return fused_chunk_simple_gla(q_f, q_f, V.expand(-1, nh, -1, -1).permute(0, 2, 1, 3).contiguous(),
                                  g, initial_state=mem, output_final_state=True, scale=1.0)


tC = t_it(c_fla, warmup=3, reps=10)

# ---- D. self 项 ----
def d_self():
    sd = (Q.float() * Q.float()).sum(-1, keepdim=True)
    return sd * V.float()


tD = t_it(d_self, warmup=2, reps=10)

print(f'  A. permute/contiguous 搬运      : {tA:8.2f} ms')
print(f'  B. 块内 raw 注意（逐块, {T//W} 块） : {tB:8.2f} ms')
print(f'  C. FLA kernel（含其输入搬运）    : {tC:8.2f} ms')
print(f'  D. self 项                      : {tD:8.2f} ms')
print()
print(f'  ⇒ 我们没换掉的 B 段 = {tB:.1f}ms 是当前最大单项！')
print(f'  ⇒ 对照：v6.7 端到端 {897.6/1000*1000:.0f}ms/step（含前向+反向+其余），'
      f'v6.6 为 {1548.4/1000*1000:.0f}ms/step')
print()
print('  【说明】以上为前向单次计时；训练还需反向（约 2-3×）。')
print('  关键判断：若 B(块内 raw 注意) 占大头 ⇒ 下一步应把块内注意也换成 FLA 的 chunk 内计算，')
print('  或让上游直接产出 [B,T,nh,N] 布局以消除 A 段搬运。')
