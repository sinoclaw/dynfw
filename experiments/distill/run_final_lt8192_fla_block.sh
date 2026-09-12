#!/bin/bash
# ② 单变量归因：v6.7 的 gate_mode 对照臂 —— 分离"实现方式"与"门控粒度"两个变量。
#
# 目的：v6.7(FLA,逐token门控) 比 v6.6(gdn,块级门控) 好 18.4%。但这两个版本差了【两个】东西：
#         ① 记忆段实现方式（顺序循环 ↔ FLA 并行）
#         ② 门控粒度（块级 ↔ 逐token）
#       本臂用 gate_mode=block 让 v6.7 精确复现 v6.6 的门控语义（块内不衰减 + 块首整体衰减），
#       从而：   实现方式差异 = v6.7(block) vs v6.6      ← 预期≈0（同语义，只是实现不同）
#                门控粒度差异 = v6.7(token) vs v6.7(block)
#
# 配置与 v6.6/v6.7 主实验完全一致（同数据/教师/seed/step/W/参数），唯一变量 = gate_mode。
# 判据（跑前锁死）：
#   主判据 = 1000 step 的 final_loss。
#   收敛判据 = 末 200 step 降幅 <1% 视为到平台（否则排名仅作趋势参考）。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 50 --snap-every 50 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 \
--cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64 \
--shared-logits /data/dynfw/results/shared_B20 --fla-gate block"

for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6.7(gate=block) seed=$sd 1000step ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_fla $D --seed $sd \
      --out "/data/dynfw/results/final_lt8192_flablock_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
echo "FLA_BLOCK_LT8192_DONE"
