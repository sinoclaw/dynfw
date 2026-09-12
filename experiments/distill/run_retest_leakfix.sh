#!/bin/bash
# 泄漏修复(d44e189)后重测 + 增量架构首次同口径对轰
# 尺子 = 批2（D=128 nh=16 L=2 mm=64 steps=1，20ep，corpus_en，共享教师 logits）
# 目的：① 重测中招架构（v6/rawfw/gdn/dla/slot_topk）② 锚点（la_cycle/bdh/tf）不变可比
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits"

# 中招架构（跨 block memory 泄漏，须重测）+ 锚点（本来就因果正确）
ARCHS="fusedfw_fw_cycle fusedfw_la_cycle fusedfw_rawfw_cycle fusedfw_gdn_cycle fusedfw_dla_cycle fusedfw_slot_topk bdh tf"

for sd in 0 1 2; do
  for a in $ARCHS; do
    EXTRA=""
    case "$a" in
      fusedfw_dla_cycle) EXTRA="--dla-k 8 --dla-w 64" ;;
      fusedfw_slot_topk) EXTRA="--slot-topk 2" ;;
    esac
    OUT=/data/dynfw/results/retest_${a}_s${sd}
    echo "=== [$(date +%H:%M:%S)] arch=$a seed=$sd -> $OUT ==="
    $PY experiments/distill/distill_qwen.py --arch "$a" $D $EXTRA --seed "$sd" --out "$OUT" 2>&1 | tail -6
    echo "--- exit=$? ---"
  done
done
echo "ALL_DONE"
