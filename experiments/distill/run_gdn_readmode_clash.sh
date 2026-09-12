#!/bin/bash
# v6.6 (gdn_cycle) 块内读侧单变量对比：softmax(原版) vs raw(对齐 v6 新默认)
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits --w 64"
for rd in softmax raw; do
  for sd in 0 1 2; do
    echo "=== [$(date +%H:%M:%S)] gdn read_mode=$rd seed=$sd ==="
    $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_cycle $D --fw-read "$rd" --seed "$sd" --out /data/dynfw/results/gdnread_${rd}_s${sd} 2>&1 | grep -E "DONE" | tail -1
  done
done
echo "GDNREAD_ALL_DONE"
