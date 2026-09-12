"""v6.7(bf16 FLA) 长 T 收口汇总：四方能力对比 + 统计检验 + 曲线。

用 stat_util.welch(a,b) -> (t, df, p)，注意返回顺序是 (t, df, p)（此前误按 (t,p,df) 解析过）。
"""
import glob
import json
import statistics as st
import sys

sys.path.insert(0, 'experiments/distill')
from stat_util import welch

PREFIX = {
    'v6.7 bf16 FLA(逐token门控)': 'final_lt8192_flabf16_s',
    'v6.7 fp32 FLA(已作废)':      'final_lt8192_fla_s',
    'v6.6 gdn(块级门控)':         'final_lt8192_gdn_s',
    'v6 fw_cycle(opt5)':         'final_lt8192_v6_s',
    'tf SDPA':                   'final_lt8192_tf_s',
}


def load(pfx):
    out = {}
    for f in sorted(glob.glob(f'results/{pfx}*/distill_result.json')):
        d = json.load(open(f))
        if d.get('final_loss') is not None:
            out[d.get('seed')] = d
    return out


G = {k: load(v) for k, v in PREFIX.items()}
vals, walls, curves = {}, {}, {}
for k, v in G.items():
    vs = [v[s]['final_loss'] for s in sorted(v)]
    vals[k] = vs
    if vs:
        walls[k] = st.median([v[s].get('wall_sec', 0) for s in sorted(v)])
        snap = {}
        for s in sorted(v):
            for x in (v[s].get('snapshots') or []):
                if isinstance(x, dict) and 'step' in x:
                    snap.setdefault(x['step'], []).append(x.get('loss_mean_recent'))
        curves[k] = {k2: st.median(v2) for k2, v2 in snap.items()}

print('=== 长 T（T=8192, 20块, 1000 step）最终 loss（越低越好）===')
for k, vs in vals.items():
    if not vs:
        print(f'  {k:26s} (无数据)')
        continue
    flag = ' [已作废: fp32 数值路径]' if 'fp32' in k else ''
    print(f'  {k:26s} n={len(vs)}  中位={st.median(vs):9.1f}  均值={st.mean(vs):9.1f}  '
          f'std={st.pstdev(vs):6.1f}  wall中位={walls.get(k,0):7.1f}s{flag}')

print()
print('=== Welch t 检验（stat_util，返回 (t, df, p)）===')
pairs = [('v6.7 bf16 FLA(逐token门控)', 'v6 fw_cycle(opt5)'),
         ('v6.7 bf16 FLA(逐token门控)', 'v6.6 gdn(块级门控)'),
         ('v6.7 bf16 FLA(逐token门控)', 'tf SDPA'),
         ('v6.7 bf16 FLA(逐token门控)', 'v6.7 fp32 FLA(已作废)')]
for a, b in pairs:
    va, vb = vals.get(a, []), vals.get(b, [])
    if len(va) < 2 or len(vb) < 2:
        print(f'  {a} vs {b}: 数据不足 ({len(va)} vs {len(vb)})')
        continue
    t, df, p = welch(va, vb)
    star = '★显著' if p < 0.05 else '不显著'
    imp = 100 * (st.median(vb) - st.median(va)) / st.median(vb)
    print(f'  {a}  vs  {b}')
    print(f'     t={t:8.2f}  df={df:5.2f}  p={p:.6f}  {star}   改善={imp:5.1f}%')

print()
print('=== 曲线中位（step: loss）===')
ks = ['v6.7 bf16 FLA(逐token门控)', 'v6.6 gdn(块级门控)', 'v6 fw_cycle(opt5)', 'tf SDPA']
steps = sorted({s for k in ks for s in curves.get(k, {})})
print(f'  {"step":>6s}' + ''.join(f'{k.split()[0]+k.split()[1]:>14s}' for k in ks))
for s in steps:
    row = f'  {s:>6d}'
    for k in ks:
        v = curves.get(k, {}).get(s)
        row += f'{v:>14.0f}' if v is not None else f'{"-":>14s}'
    print(row)
print()
nonconv = []
for k in ks:
    c = curves.get(k, {})
    if len([s for s in c if s in (800, 1000)]) == 2:
        drop = 100 * (c[800] - c[1000]) / c[800]
        nonconv.append((k, drop))
print('=== 收敛判据（末 200 step 降幅 <1% 视为到平台）===')
for k, d in nonconv:
    print(f'  {k:26s} 降幅 {d:5.1f}%  {"到平台 ✓" if d < 1 else "仍在降 ✗"}')
