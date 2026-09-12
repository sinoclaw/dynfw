#!/bin/bash
# 串行执行：① 扫 T（v6.7+compile vs v6+opt5 vs TF） ② 长 T 重跑验证能力不损（J2）
# 单卡串行，避免互相干扰。
cd /data/dynfw || exit 1
PY=/data/dynfw-env/bin/python
export PYTHONPATH=/data/dynfw

echo "=== [$(date +%H:%M:%S)] 预计阶段 ①：扫 T ==="
$PY benchmarks/bench_sweep_T_v67c_vs_tf.py 2>&1

echo
echo "=== [$(date +%H:%M:%S)] 阶段 ① 结束，开始 ②：长 T 重跑（J2 能力不损验证）==="
echo "参照值：v6.7 升级前 1000step 长T 中位 5738.0（s0 5780.5 / s1 5738.0 / s2 5733.2）"
D="--data-bin /data/dynfw/data/tinystories_qwen.bin --block 8192 --chunk 2048 --batch 1 \
--max-batches 20 --epochs 50 --snap-every 50 --dim 128 --nh 16 --n-layer 2 --mlp-mult 64 \
--cycle-steps 1 --teacher /data/models/Qwen3-0.6B --teacher-half --w 64 \
--shared-logits /data/dynfw/results/shared_B20"
for sd in 0 1 2; do
  echo "=== [$(date +%H:%M:%S)] v6.7 seed=$sd（升级后 torch 2.7.1）==="
  $PY experiments/distill/distill_qwen.py --arch fusedfw_gdn_fla $D --seed $sd \
      --out "/data/dynfw/results/post_upgrade_v67_s${sd}" 2>&1 | grep -E "DONE|Error|Traceback" | tail -1
done
echo "POST_UPGRADE_DONE"
