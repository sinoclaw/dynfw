#!/bin/bash
# v6 块内读侧单变量对比：softmax(原版) vs raw(对齐 BDH-CQ 官方)
# 口径与 W=64 公平矩阵一致（D=128 nh=16 L=2 mm=64 W=64，corpus_en，3 seed）
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits --w 64"

for rd in softmax raw; do
  for sd in 0 1 2; do
    OUT=/data/dynfw/results/readmode_${rd}_s${sd}
    echo "=== [$(date +%H:%M:%S)] v6 read_mode=$rd seed=$sd ==="
    $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle $D --fw-read "$rd" --seed "$sd" --out "$OUT" 2>&1 | grep -E "DONE" | tail -1
  done
done
echo "READMODE_ALL_DONE"
