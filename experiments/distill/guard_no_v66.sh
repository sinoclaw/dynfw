#!/bin/bash
# 守卫：确保 v6.6 不会被跑（2026-09-13 爸爸指示停掉 v6.6）
# 双重保险 —— 即使 bash 缓冲了旧版 run_conv3000.sh，本守卫也会终止 v6.6 进程。
# 注意：本脚本以文件形式运行，pgrep 模式不出现在自身命令行中，避免 pgrep -f 自匹配自杀。
LOG=/data/dynfw/results/conv3000_log.txt
PAT="[d]istill_qwen.py --arch fusedfw_gdn "   # [d] 字符类技巧：不匹配本行字面量
killed=0
while true; do
  PIDS=$(pgrep -f "$PAT" 2>/dev/null)
  if [ -n "$PIDS" ]; then
    echo "[$(date +%H:%M:%S)] 守卫: 检出 v6.6 进程 → 终止: $(echo $PIDS | tr '\n' ' ')"
    kill -9 $PIDS 2>/dev/null
    killed=$((killed+1))
    echo "!! 注意：说明 bash 读的是旧脚本，v6.6 已被阻止 $killed 次" >> $LOG
  fi
  if grep -q "CONV3000_DONE" $LOG 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] 守卫: 完成标志已出现（v6.6 拦截 $killed 次），退出"
    break
  fi
  if ! pgrep -f "[r]un_conv3000.sh" >/dev/null 2>&1 && ! pgrep -f "[c]onv3000_v6" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] 守卫: 主脚本已结束（v6.6 拦截 $killed 次），退出"
    break
  fi
  sleep 60
done
