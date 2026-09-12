"""列出 results 下指定前缀的产物（arch/seed/loss/wall/epochs）。用法: python ls_results.py lt8192_ lt8192opt5_ ..."""
import glob
import json
import os
import sys

R = '/data/dynfw/results'
prefixes = sys.argv[1:] or ['lt8192']
rows = []
for p in prefixes:
    for d in sorted(glob.glob(os.path.join(R, p + '*'))):
        if not os.path.isdir(d):
            continue
        f = os.path.join(d, 'distill_result.json')
        name = os.path.basename(d)
        if not os.path.exists(f):
            rows.append((p, name, None))
            continue
        j = json.load(open(f))
        if name.startswith('smoke'):
            continue
        rows.append((p, name, j))

for p in prefixes:
    print(f'--- 前缀 {p} ---')
    for _, name, j in rows:
        if not name.startswith(p):
            continue
        if j is None:
            print(f'  {name:30s} (未完成)')
        else:
            print(f'  {name:30s} arch={j.get("arch"):22s} seed={j.get("seed")} '
                  f'final_loss={j.get("final_loss"):12.4f} wall={j.get("wall_sec"):7.1f}s '
                  f'epochs={j.get("epochs")} T={j.get("block")}')
    print()
