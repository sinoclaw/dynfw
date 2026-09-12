#!/bin/bash
# 真实预训练冒烟: 各架构 next-token 预训练能否收敛 (小规模, 快速)
cd /data/dynfw
export PYTHONPATH=/data/dynfw
PY=/data/dynfw-env/bin/python
D="--data /data/dynfw/data/wikitext103.txt --teacher /data/models/Qwen3-0.6B --dim 64 --nh 8 --n-layer 1 --mlp-mult 32 --block 64 --batch 4 --max-batches 6 --epochs 2 --max-lines 300"
for a in v6 tf bdh la_cycle dla; do
  echo "===== $a ====="
  $PY experiments/distill/pretrain_clash.py --arch $a $D --out /data/dynfw/results/pretrain_smoke_$a 2>&1 | tail -6
done
echo SMOKE_DONE
