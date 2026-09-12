"""8-seed 三架构对照（v5 精确 vs v6 压缩 vs gdn 门控），raw 单变量。"""
import glob
import json
import os
import statistics

R = '/data/dynfw/results'


def vals(pats):
    out = []
    for p in pats:
        for f in glob.glob(p, recursive=True):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if isinstance(j, dict) and 'final_loss' in j:
                out.append(j['final_loss'])
    return sorted(out)


def show(name, d):
    if not d:
        print(f'{name:16s} (空)')
        return
    print(f'{name:16s} n={len(d)}  中位={statistics.median(d):7.2f}  均值={statistics.mean(d):7.2f}  '
          f'std={statistics.stdev(d) if len(d) > 1 else 0:5.2f}  min={min(d):6.2f} max={max(d):6.2f}')
    print(f'                 {[round(x, 2) for x in d]}')


print('=== raw 口径（W=64）8 seed 三架构对照 ===\n')
v5 = vals([f'{R}/retest_fusedfw_la_cycle_s*/**/*.json', f'{R}/v5anchor_s*/**/*.json'])
show('v5 la_cycle', v5)
print()
show('v6 raw', vals([f'{R}/readmode_raw_s*/**/*.json']))
print()
show('gdn raw', vals([f'{R}/gdnread_raw_s*/**/*.json']))
print()
print('--- 对照（softmax 旧口径，3 seed）---')
show('v6 softmax', vals([f'{R}/readmode_softmax_s*/**/*.json']))
show('gdn softmax', vals([f'{R}/gdnread_softmax_s*/**/*.json']))
print()
print('--- 外部对手锚（3 seed）---')
show('bdh', vals([f'{R}/retest_bdh_s*/**/*.json']))
show('tf', vals([f'{R}/retest_tf_s*/**/*.json']))
