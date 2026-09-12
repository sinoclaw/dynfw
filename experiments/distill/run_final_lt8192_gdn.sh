#!/bin/bash
# v6.6(gdn_cycle) 长 T 实验 —— 补上门控在长序列的长跑数据（门控理论主场）。
#
# 与 v6 收口实验**同配置**（同数据/同教师/同 seed/同 step），唯一变量 = arch：
#   T=8192 / 20块 / 50ep = 1000step / snap 每 50 step / 3 seed / 复用 shared_B20 logits
#
# 形态说明（诚实标注）：v6 用 opt5_raw（交付形态），v6.6 用**基线形态**
#   —— 因为 to_opt5_raw 目前只支持 v6 的 FWAttention 类，v6.6 的 GDNFastAttn 是另一个类。
#   能力对比**不受形态影响**（J3 已验 opt5 与基线等价：Δ 与 seed 波动同量级）；
#   速度**不作对比**（形态不同）。
#
# 判据（跑前锁死）：
#   主判据 = 1000 step 时 v6.6 与 v6 的 loss 中位差；Welch t-test（3 seed）。
#   收敛判据 = 末 200 step 降幅 <1% 视为到平台。
#   预期：门控价值只有在长 T 才可能显现（v6 的 ‖M‖ 到 T=8192 已达 1423、v6.6 为 677）。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 50 --snap-every 50 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 \
--cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64 \
--shared-logits /data/dynfw/results/shared_B20"

for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6.6 seed=$sd 1000step ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_cycle $D --seed $sd \
      --out "/data/dynfw/results/final_lt8192_gdn_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
echo "GDN_LT8192_DONE"
