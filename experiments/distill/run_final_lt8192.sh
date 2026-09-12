#!/bin/bash
# 收口实验：v6(opt5 交付形态) vs TF，长 T=8192，训到 1000 step，3 seed。
#
# ── 判据（跑前锁死，事后不改）────────────────────────────────
#  主判据 : 1000 step 时的 loss（snapshots 中 step=1000 的 loss_mean_recent）中位，
#           v6 < tf 记为「v6 在长 T 胜」，并做 Welch t-test（3 seed）。
#  收敛判据: 末 200 step 的 loss 下降 < 1% ⇒ 视为已到平台；否则只报「同预算对比」。
#  口径    : 同数据(tinystories_qwen.bin 前 20 块)/同教师/同 seed/同优化等级(opt5_raw vs SDPA)/
#           同 step(1000)/同 batch(1)。参数量差异 45.19M vs 39.44M 如实标注。
#  预算    : 20 块 × 50 epoch = 1000 step（snap 每 50 step）
# ─────────────────────────────────────────────────────────────
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
BASE="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 50 --snap-every 50 \
--dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B \
--teacher-half --w 64 --shared-logits /data/dynfw/results/shared_B20"

for sd in 0 1 2; do
  for arch in fusedfw_fw_cycle tf; do
    if [ "$arch" = "tf" ]; then tag=tf; EXTRA=""; else tag=v6; EXTRA="--opt5"; fi
    echo "=== [$(date +%H:%M:%S)] seed=$sd $tag 1000step ==="
    $PY experiments/distill/distill_qwen.py --arch "$arch" $EXTRA $BASE --seed $sd \
        --out "/data/dynfw/results/final_lt8192_${tag}_s${sd}" 2>&1 \
      | grep -E "DONE|Error|Traceback" | tail -1
  done
done
echo "FINAL_LT8192_ALL_DONE"
