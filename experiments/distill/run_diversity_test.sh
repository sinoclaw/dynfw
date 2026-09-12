#!/bin/bash
# 隔离「样本多样性」变量：验证长 T 的落后是不是「小样本拟合速度」的假象。
#
# 三臂，**总优化步数相同**（200 step），唯一变量 = 输入多少段不同的文本：
#   A  5 块 × 20 epoch = 100 step   （已有数据，参考）
#   B 40 块 ×  5 epoch = 200 step   （多样性 8×）
#   C  5 块 × 40 epoch = 200 step   （同 step 但多样性不变 ⇒ 隔离变量）
#
# 判读：
#   v6 在 B 上相对 TF 的劣势 明显小于 在 C 上 ⇒ 「输」主要来自样本多样性/小样本拟合
#   B 上仍显著落后 ⇒ 长上下文能力真短板
#
# 口径：T=8192, W=64, chunk 2048, teacher-half，数据 tinystories_qwen.bin（同源）
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
BASE="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B \
--teacher-half --w 64 --seed 0"

run () {  # $1=arch $2=extra $3=outname $4=label
  echo "=== [$(date +%H:%M:%S)] $4 ==="
  SH="/data/dynfw/results/shared_$(echo $3 | grep -o "B20\|C5")"
  $PY experiments/distill/distill_qwen.py --arch "$1" $2 $BASE --shared-logits "$SH" --out "/data/dynfw/results/$3" 2>&1 \
    | grep -E "DONE|Error|Traceback" | tail -1
}

for arch in fusedfw_fw_cycle tf; do
  if [ "$arch" = "tf" ]; then tag=tf; EXTRA=""; else tag=v6; EXTRA="--opt5"; fi
  run "$arch" "$EXTRA --max-batches 20 --epochs 10"  "diversity_${tag}_B20" "B: $tag, 20块x10ep (200step, 多样性4x)"
  run "$arch" "$EXTRA --max-batches 5  --epochs 40" "diversity_${tag}_C5"  "C: $tag, 5块x40ep (200step, 多样性1x)"
done
echo "DIVERSITY_ALL_DONE"
