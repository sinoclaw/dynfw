#!/bin/bash
# 加步数收敛实验（2026-09-13）—— 把主线三方从 1000 step 拉到 3000 step，检验排名是否守住。
#
# 背景：1000 step 时末 200 step 仍降 24.5%（远未到平台），故 1000-step 的排名只是趋势。
#   本实验把 step 数 ×3，判据（跑前锁死，不事后改）：
#     J1 排名是否守住：v6.7 < v6 < v6.6（1000 step 的三方顺序不变）
#     J2 收敛判据：末 200 step 降幅 <1% 视为到平台（<1% 则本实验可定终局排名）
#     J3 差距是否扩大/缩小：记录每方相对 v6 的相对改善百分比变化
#     J4 诚实条款：若仍不收敛，只报"趋势持续/反转"，不报终局
#
# 配置与 1000-step 实验**完全一致**，唯一变量 = --epochs（50 → 150，即 1000 → 3000 step）。
#   复用同一份 shared_B20 教师 logits（同源、无新 IO 变量）。
#   判据跑前锁死，事后不改。
#
# 不修的东西（明确声明）：本实验不解决显存问题、不做新架构、不改数据。只回答"排名是否守住"。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 150 --snap-every 50 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 \
--cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64 \
--shared-logits /data/dynfw/results/shared_B20"

for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6.7(FLA,token) seed=$sd 3000step ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_fla $D --seed $sd \
      --out "/data/dynfw/results/conv3000_v67_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6(raw opt5) seed=$sd 3000step ==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_fw_cycle $D --seed $sd --opt5 \
      --out "/data/dynfw/results/conv3000_v6_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
# --- v6.6 段已按爸爸指示移除（2026-09-13 03:2x：v6.6 不再作为对照跑）---
# for sd in 0 1 2; do
#   echo "=== [$(date +%H:%M:%S)] v6.6(gdn) seed=$sd 3000step ==="
#   $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn $D --seed $sd \
#       --out "/data/dynfw/results/conv3000_v66_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
# done
echo "CONV3000_DONE"
