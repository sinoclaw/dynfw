#!/bin/bash
# 补 seed 3-7：三架构同口径扩展（gdn raw / v6 raw / v5 la_cycle）
# 目的：gdn(raw) 方差大(10.76) vs v5(1.62)，3 seed 不足以定论 → 扩到 8 seed
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits --w 64"

for sd in 3 4 5 6 7; do
  echo "=== [$(date +%H:%M:%S)] seed=$sd ==="
  echo "--- gdn(raw) ---"
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_cycle $D --fw-read raw --seed $sd --out /data/dynfw/results/gdnread_raw_s${sd} 2>&1 | grep -E "DONE" | tail -1
  echo "--- v6(raw) ---"
  $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle $D --fw-read raw --seed $sd --out /data/dynfw/results/readmode_raw_s${sd} 2>&1 | grep -E "DONE" | tail -1
  echo "--- v5(la_cycle) ---"
  $PY experiments/distill/distill_qwen.py --arch fusedfw_la_cycle $D --seed $sd --out /data/dynfw/results/v5anchor_s${sd} 2>&1 | grep -E "DONE" | tail -1
done
echo "SEED_EXTEND_ALL_DONE"
