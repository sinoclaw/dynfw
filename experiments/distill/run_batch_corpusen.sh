#!/bin/bash
cd /data/dynfw
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits"
for sd in 0 1 2; do
  $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle $D --seed $sd --out /data/dynfw/results/v6fix_corpusen_s$sd
  $PY experiments/distill/distill_qwen.py --arch bdh              $D --seed $sd --out /data/dynfw/results/bdh_corpusen_s$sd
done
$PY experiments/distill/distill_qwen.py --arch bdh_rawfw_qwen  $D --seed 0 --out /data/dynfw/results/rawfw_corpusen_s0
echo DONE_BATCH
