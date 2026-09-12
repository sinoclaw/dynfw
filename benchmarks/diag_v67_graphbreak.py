"""查 v6.7 + torch.compile 的 graph break —— 精确定位断点。

判据（跑前锁死，事后不改）：
  J1 能否减少 break 数量 / 能否让 FLA 段被当作一个 opaque op
  J2 速度：目标 ≤ 56.9ms（打平 v6+opt5）；若 >70ms 视为无改善
  J3 数值：改造后 logits 与改造前 maxdiff 应 ≤ bf16 量级（FLA 段本就是 bf16）
  J4 诚实：若无法改善，如实记录，不硬凑
"""
import sys, time, statistics
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
sys.setrecursionlimit(10000)
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
torch.backends.cuda.matmul.allow_tf32 = True

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM

m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                  mlp_mult=MLP_MULT, W=W, read_mode='raw').to(DEV)
x = torch.randint(0, 1000, (B, T), device=DEV)

print("=" * 88)
print("① torch._dynamo.explain —— 看 graph break 数量与位置")
print("=" * 88)
try:
    import torch._dynamo as dynamo
    exp = dynamo.explain(m.forward_hidden)(x)
    print(f"  图数量 (graph_count):     {exp.graph_count}")
    print(f"  断点数量 (graph_break_count): {len(exp.break_reasons)}")
    print(f"  操作数 (op_count):        {exp.op_count}")
    print()
    print("  --- 断点原因（按出现次数）---")
    seen = {}
    for br in exp.break_reasons:
        r = str(br.reason)[:150]
        seen[r] = seen.get(r, 0) + 1
    for r, n in sorted(seen.items(), key=lambda t: -t[1])[:12]:
        print(f"    ×{n:3d}  {r}")
    print()
    print("  --- 涉及的用户代码位置 ---")
    locs = {}
    for br in exp.break_reasons:
        for f in br.user_stack or []:
            k = f.split('/data/dynfw/')[-1].split(' (')[0]
            locs[k] = locs.get(k, 0) + 1
    for k, n in sorted(locs.items(), key=lambda t: -t[1])[:10]:
        print(f"    ×{n:3d}  {k}")
except Exception as e:
    import traceback
    print(f"  explain 失败: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
