#!/bin/bash
# 长 T（8192）**交付形态**能力对轰：opt5_raw(DynFW 交付形态) vs TF
#   外加基线版 fw_cycle 做正式 J3（训练等价性）对照。
#
# 预算：按 token 数对齐 8 seed 基线（800 step × 1024 token = 819,200）
#       ⇒ T=8192 时 100 样本 = max-batches 5 × epochs 20
# 口径：D=128 nh=16 L=2 mm=64, W=64, --chunk 2048, --teacher-half,
#       数据 = tinystories(教师同词表重tokenize)
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 --max-batches 5 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64"

for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] seed=$sd : opt5_raw(v6) ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle --opt5 $D --seed $sd \
      --out /data/dynfw/results/lt8192opt5_fw_s${sd} 2>&1 | grep -E "DONE|Error" | tail -1

  echo "=== [$(date +%H:%M:%S)] seed=$sd : baseline(v6, 无 opt5) = J3 对照 ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle $D --seed $sd \
      --out /data/dynfw/results/lt8192base_fw_s${sd} 2>&1 | grep -E "DONE|Error" | tail -1

  echo "=== [$(date +%H:%M:%S)] seed=$sd : TF ==="
  $PY experiments/distill/distill_qwen.py --arch tf $D --seed $sd \
      --out /data/dynfw/results/lt8192opt5_tf_s${sd} 2>&1 | grep -E "DONE|Error" | tail -1
done
echo "LT8192_OPT5_ALL_DONE"
