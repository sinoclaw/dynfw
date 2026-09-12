"""汇总 /data/dynfw/results 下所有蒸馏产物为可比较的表格（按架构+read_mode 分组统计）。"""
import glob
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../results'
ROOT = '/data/dynfw/results'

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, '*'))):
    if not os.path.isdir(d):
        continue
    j = None
    for f in glob.glob(os.path.join(d, '**', '*.json'), recursive=True):
        try:
            cand = json.load(open(f))
        except Exception:
            continue
        if isinstance(cand, dict) and 'final_loss' in cand:
            j = cand
            break
    if j is None:
        continue
    name = os.path.basename(d)
    seed = None
    m = re.search(r'_s(\d+)$', name)
    if m:
        seed = int(m.group(1))
    rows.append({'dir': name, 'seed': seed,
                 'arch': j.get('arch', name), 'loss': j['final_loss'],
                 'w': j.get('block', None), 'params': j.get('student_params')})

print(f"共 {len(rows)} 个产物\n")
print(f"{'目录':40s} {'架构':24s} {'seed':>4s} {'final_loss':>12s}")
print('-' * 86)
for r in sorted(rows, key=lambda x: (x['arch'], x['seed'] if x['seed'] is not None else -1)):
    print(f"{r['dir']:40s} {r['arch']:24s} {str(r['seed']):>4s} {r['loss']:12.4f}")

# 按 (arch, 目录前缀) 分组统计
groups = {}
for r in rows:
    key = r['dir']
    key = re.sub(r'_s\d+$', '', key)
    groups.setdefault(key, []).append(r['loss'])

print(f"\n{'组':40s} {'n':>3s} {'中位':>10s} {'均值':>10s} {'std':>9s} {'最小值':>10s} {'最大值':>10s}")
print('-' * 95)
for k in sorted(groups):
    v = groups[k]
    if len(v) < 2:
        continue
    print(f"{k:40s} {len(v):3d} {statistics.median(v):10.2f} {statistics.mean(v):10.2f} "
          f"{(statistics.stdev(v) if len(v) > 1 else 0):9.2f} {min(v):10.2f} {max(v):10.2f}")
