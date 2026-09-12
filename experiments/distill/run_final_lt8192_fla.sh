#!/bin/bash
# v6.7(FLA) 长 T 实验 —— 把 v6.6 的门控记忆段（顺序循环）换成 FLA 的 fused_chunk_simple_gla。
#
# 与 v6.6 实验**同配置**（同数据/同教师/同 seed/同 step/同 W/同参数），唯一变量 = 记忆段的实现。
#   T=8192 / 20块 / 50ep = 1000step / snap 每 50 step / 3 seed / 复用 shared_B20 logits
#
# 诚实标注（重要，不可省）：
#   ⚠️ FLA 是【逐 token 门控】，v6.6 是【块级门控】（alpha.mean(dim=2)）。
#   两者数学上不可精确互转 ⇒ 本实验【同时】改变了两个变量：①记忆段实现方式 ②门控粒度。
#   因此：
#     - 速度对比(vs v6.6)：只反映实现方式差异（门控粒度对 wall 影响极小）
#     - 能力对比(vs v6.6)：是"实现+粒度"的合成效应，需再用 gate_mode=block 跑一臂做单变量拆解
#   本脚本默认 gate_mode=token（FLA 原生）。
#
# 判据（跑前锁死）：
#   主判据 = wall（vs v6.6 的 1548s）；次判据 = 1000 step 的 loss（vs v6.6 的 7142.6 中位）。
#   收敛判据 = 末 200 step 降幅 <1% 视为到平台。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 50 --snap-every 50 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 \
--cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64 \
--shared-logits /data/dynfw/results/shared_B20"

for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6.7(FLA) seed=$sd 1000step ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_fla $D --seed $sd \
      --out "/data/dynfw/results/final_lt8192_fla_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
echo "FLA_LT8192_DONE"
