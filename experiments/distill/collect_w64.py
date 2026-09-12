"""W=block vs W=64 两批矩阵并排汇总（含一致性检查）。"""
import glob
import json
import statistics
from collections import defaultdict


def load(pattern):
    rows = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
            rows[d['arch']].append((d['seed'], float(d['final_loss'])))
        except Exception as e:
            print('bad', f, e)
    return rows


w256 = load('/data/dynfw/results/retest_*/distill_result.json')
w64 = load('/data/dynfw/results/w64_*/distill_result.json')


def med(v):
    return statistics.median([x[1] for x in v]) if v else float('nan')


print('=' * 78)
print('W=64 公平版排名（所有分块架构统一窗口宽；中位 KL，越低越好）')
print('=' * 78)
for a, v in sorted(w64.items(), key=lambda kv: med(kv[1])):
    detail = '  '.join(f's{s}={l:.2f}' for s, l in sorted(v))
    print(f'{a:26s} 中位={med(v):8.2f}   {detail}')

print()
print('=' * 78)
print('W=block(256) vs W=64 对照（Δ<0 = 缩小窗口后变好）')
print('=' * 78)
print(f"{'架构':28s} {'W=block':>10s} {'W=64':>10s} {'Δ':>10s}")
for a in sorted(set(w256) | set(w64)):
    a256, a64 = med(w256.get(a, [])), med(w64.get(a, []))
    line = f'{a:28s} {a256:10.2f} {a64:10.2f}'
    if a in w256 and a in w64:
        line += f' {a64 - a256:+10.2f}'
    else:
        line += f" {'(该批无)':>10s}"
    print(line)

print()
print('注：v7 dla 在 W=block 批里已用 --dla-w 64（即 W=64），故两批应接近 → 可作一致性检查。')
