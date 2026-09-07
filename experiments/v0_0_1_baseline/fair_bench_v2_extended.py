"""FAIR_BENCH_V2 (extended) —— 补基准覆盖：batch{1,8,32} × T{256,1024,4096,8192} prefill 矩阵 + decode@batch1。

v0.0.1 只跑了 prefill batch=8、T 到 4096；本脚本把 batch 与 T 两个轴钉全。
复用 v0.0.1 的 ~400K 参数匹配模型（TF_sdpa / FusedFW+FFN），起计时（wall-clock）。
GPU 列（FlashAttention）需有卡本地无 GPU → 标 \"待补\"，不假装测过。

注：TF_sdpa 用 SDPA(is_causal) 现代基线；无 KV-cache 的 prefill 是全序列一次前向。
decode 是自回归，batch 天然 = 1（逐 token），故只报 batch1 每 token 毫秒。
"""
import sys, time, torch, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
torch.set_num_threads(8)

from dynfw.models.fused_fw import FusedFW
from fair_bench_v2 import TF_sdpa, search_width, mk_tf, mk_fw, prefill_time, decode_tf_timing, decode_fw_timing

TARGET = 400_000
dtf = search_width(mk_tf, TARGET); dfw = search_width(mk_fw, TARGET)
mtf = mk_tf(dtf[0]); mfw = mk_fw(dfw[0])
print(f"=== FAIR_BENCH_V2 extended (CPU) — batch x seq prefill 矩阵 ===", flush=True)
print(f"  匹配 {TARGET:,} 参数 → D_tf={dtf[0]}({dtf[1]:,}) D_fw={dfw[0]}({dfw[1]:,})", flush=True)


def prefill_reps(T, B):
    # 大 T / 大 batch 用少 reps 控制总时长
    if T >= 4096: return 1
    return 2


grid = []
print(f"{'batch':>5} {'T':>6} {'TF ms':>10} {'FW ms':>10} {'FW/TF':>7}  {'备注'}", flush=True)
for B in [1, 8, 32]:
    for T in [256, 1024, 4096, 8192]:
        xt = torch.randint(0, 256, (B, T))
        reps = prefill_reps(T, B)
        a = b = float('nan')
        try:
            a = prefill_time(mtf, xt, warmup=1, reps=reps)
            b = prefill_time(mfw, xt, warmup=1, reps=reps)
            ratio = b / max(a, 1e-9)
            note = ""
            grid.append(dict(batch=B, T=T, tf_ms=float(a*1000), fw_ms=float(b*1000), fw_over_tf=float(ratio)))
        except Exception as e:
            ratio = float('nan'); note = f"SKIP {e}"
        print(f"{B:>5} {T:>6} {a*1000:>10.1f} {b*1000:>10.1f} {ratio:>7.3f}  {note}", flush=True)

# decode @ batch1（自回归逐 token）
dtok = decode_tf_timing(mtf, torch.zeros((1, 256), dtype=torch.long))
dfok = decode_fw_timing(mfw)
print(f"\ndecode@batch1 (每 token ms): TF={dtok*1000:.2f} FW={dfok*1000:.2f}  FW/TF={dfok/max(dtok,1e-9):.2f}x", flush=True)

import json
json.dump(dict(target=TARGET, D_tf=dtf[0], D_fw=dfw[0],
               prefill=grid, decode=dict(tf_ms=dtok*1000, fw_ms=dfok*1000,
                                          fw_over_tf=dfok/max(dtok,1e-9))),
          open('results/v0.0.1/fair_bench_v2_extended.json', 'w'), indent=2)
print("\nDONE saved results/v0.0.1/fair_bench_v2_extended.json", flush=True)
