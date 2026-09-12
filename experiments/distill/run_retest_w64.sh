#!/bin/bash
# 公平口径重跑：所有「分块 + 跨块/跨 chunk 记忆」架构统一 W=64
# 理由：泄漏修复(d44e189)后 memory 按 block 独立、层内 chunk 间累积 →
#       若 W=block(256) 则整层仅 1 个 chunk，跨块记忆等于被关闭（v6/v6.6/v8 退化到 354）。
#       要让「压缩状态」机制真被触发，W 必须 < block（本例 256/64 = 4 个 chunk）。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--block 256 --batch 4 --max-batches 40 --epochs 20 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 --cycle-steps 1 --teacher /data/models/Qwen3-0.6B --data /data/dynfw/data/corpus_en.txt --shared-logits /data/dynfw/results/shared_logits"
ARCHS="fusedfw_fw_cycle fusedfw_gdn_cycle fusedfw_slot_topk fusedfw_dla_cycle"

for sd in 0 1 2; do
  for a in $ARCHS; do
    EXTRA=""
    case "$a" in
      fusedfw_dla_cycle) EXTRA="--dla-k 8" ;;
      fusedfw_slot_topk) EXTRA="--slot-topk 2 --dla-k 8" ;;
    esac
    OUT=/data/dynfw/results/w64_${a}_s${sd}
    echo "=== [$(date +%H:%M:%S)] W=64 arch=$a seed=$sd -> $OUT ==="
    $PY experiments/distill/distill_qwen.py --arch "$a" $D --w 64 $EXTRA --seed "$sd" --out "$OUT" 2>&1 | tail -4
    echo "--- exit=$? ---"
  done
done
echo "W64_ALL_DONE"