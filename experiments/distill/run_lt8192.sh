#!/bin/bash
# 长 T（8192）能力对轰：压缩 vs 精确 vs 门控 vs TF
#
# 预算对齐方式 = **按处理 token 数对齐**（今天的 8 seed 基线：block256 batch4 40batch 20ep
#   = 800 step × 1024 token = 819,200 token）。
#   长 T：8192 token/样本 ⇒ 819,200/8192 = 100 样本 ⇒ max-batches 5 × epochs 20 = 100 step。
# 口径：D=128 nh=16 L=2 mm=64，W=64（分块架构统一，铁律 0），--chunk 2048，--teacher-half，
#       数据 = tinystories 用教师同词表重新 tokenize（非 corpus_en ⇒ 与 8 seed 那套数值不可直接比，
#       架构之间可比）。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 --max-batches 5 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64"

for sd in 0 1 2; do
    OUT=/data/dynfw/results/lt8192_${arch}_s${sd}
    echo "=== [$(date +%H:%M:%S)] T=8192 arch=$arch seed=$sd ==="
    $PY experiments/distill/distill_qwen.py --arch $arch $D --seed $sd --out "$OUT" 2>&1 | grep -E "DONE|Error|error" | tail -1
  done
done
echo "LT8192_ALL_DONE"
