#!/bin/bash
# 回滚：torch 2.7.1 → 2.5.1（若验收 J1/J4/J5 不通过）
# 快照：docs/env-snapshot-pre-torch-upgrade-20260913.txt（81 个包）
PY=/data/dynfw-env/bin/python
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [$(date +%H:%M:%S)] 当前 ==="
$PY -c "import torch, triton; print(f'  torch={torch.__version__} triton={triton.__version__}')"

echo
echo "=== 回滚 torch 2.5.1 + triton 3.1.0 ==="
$PY -m pip install -i $MIRROR --no-cache-dir "torch==2.5.1" "triton==3.1.0" 2>&1 | tail -12

echo
echo "=== 回滚后 ==="
$PY -c "
import torch, triton
print(f'  torch={torch.__version__} (cuda {torch.version.cuda}) triton={triton.__version__}')
print(f'  cuda: {torch.cuda.is_available()}')
"
echo "ROLLBACK_DONE"
