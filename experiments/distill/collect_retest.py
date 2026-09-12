"""汇总泄漏修复后的重测矩阵：按 arch 分组、算中位数、排名。"""
import glob
import json
import statistics
from collections import defaultdict

rows = []
for f in sorted(glob.glob('/data/dynfw/results/retest_*/distill_result.json')):
    try:
        d = json.load(open(f))
        d['_file'] = f
        rows.append(d)
    except Exception as e:
        print('读取失败', f, e)

print(f'共收集 {len(rows)} 个 run\n')
g = defaultdict(list)
for d in rows:
    g[d.get('arch', '?')].append((d.get('seed', -1), float(d.get('final_loss', float('nan')))))

print(f"{'架构':28s} {'seed数':>5s} {'中位KL':>10s} {'min':>9s} {'max':>9s}  逐seed")
print('-' * 88)
ranked = sorted(g.items(), key=lambda kv: statistics.median([x[1] for x in kv[1]]))
for a, v in ranked:
    losses = [x[1] for x in v]
    seeds = sorted(x[0] for x in v)
    med = statistics.median(losses)
    detail = '  '.join(f's{s}={l:.2f}' for s, l in sorted(v))
    print(f'{a:28s} {len(v):5d} {med:10.2f} {min(losses):9.2f} {max(losses):9.2f}  {detail}')

print('\n=== 中位数排名（越低越好）===')
for i, (a, v) in enumerate(ranked, 1):
    print(f'  {i}. {a:28s} {statistics.median([x[1] for x in v]):.2f}')

# 锚点对照（批2 泄漏版 → 本次修复后）
anchors = {'fusedfw_la_cycle': 84.57, 'bdh': 88.41, 'tf': 168.98, 'fusedfw_fw_cycle': 82.39}
print('\n=== 与批2（泄漏版）对照 ===')
for a, old in anchors.items():
    if a in g:
        new = statistics.median([x[1] for x in g[a]])
        print(f'  {a:26s} 泄漏版={old:8.2f}  修复后={new:8.2f}  变化={new-old:+.2f}')
