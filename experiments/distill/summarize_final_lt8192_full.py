"""长 T(8192) 三方完整汇总：v6(opt5) / v6.6(基线形态) / tf(SDPA)，3 seed，1000 step。

判据（跑前锁死）：
  主判据 = 1000step 时 loss 中位 + Welch t-test（3 seed）；
  收敛判据 = 末 200 step 降幅 <1% 视为到平台；
  口径标注 = v6 交付形态 opt5_raw，v6.6 基线形态（to_opt5_raw 目前只支持 v6），tf SDPA；
             参数量 v6=45,187,072 / v6.6=45,188,098 / tf=39,444,352。
"""
import glob
import json
import math
import os
import statistics

R = '/data/dynfw/results'


def load(tag):
    out = {}
    for d in sorted(glob.glob(os.path.join(R, f'final_lt8192_{tag}_s*'))):
        f = os.path.join(d, 'distill_result.json')
        if os.path.exists(f):
            j = json.load(open(f))
            out[j['seed']] = j
    return out


def welch(a, b):
    """用自带 stdlib 实现（GPU venv 无 scipy，pip 装会超时）—— 已与标准临界值自检一致。"""
    import sys as _s
    _s.path.insert(0, '/data/dynfw/experiments/distill')
    from stat_util import welch as _w
    return _w(a, b)


D = {t: load(t) for t in ('v6', 'gdn', 'tf')}
names = {'v6': 'v6 fw_cycle (opt5)', 'gdn': 'v6.6 gdn_cycle (基线形态)', 'tf': 'tf SDPA'}

print('=== 最终 loss（1000 step）===')
fin = {}
for k in ('v6', 'gdn', 'tf'):
    d = D[k]
    if not d:
        continue
    vs = [d[s]['final_loss'] for s in sorted(d)]
    fin[k] = vs
    print(f'  {names[k]:28s} n={len(vs)}  中位={statistics.median(vs):10.1f}  '
          f'均值={statistics.mean(vs):10.1f}  std={statistics.stdev(vs) if len(vs)>1 else 0:7.1f}  {[round(v,1) for v in vs]}')

print('\n=== 两两对比（Welch t-test）===')
for a, b in (('gdn', 'v6'), ('v6', 'tf'), ('gdn', 'tf')):
    if a in fin and b in fin:
        t, df, p = welch(fin[a], fin[b])
        ma, mb = statistics.median(fin[a]), statistics.median(fin[b])
        better = names[a].split()[0] if ma < mb else names[b].split()[0]
        print(f'  {names[a]:28s} vs {names[b]:22s}  中位 {ma:.1f} vs {mb:.1f} '
              f'(胜={better}, {(max(ma,mb)/min(ma,mb)-1)*100:.1f}%)  Welch t={t:+.2f} p={p:.4f} df={df:.1f}')

print('\n=== 各 step 的中位 loss 曲线 ===')
steps = [100, 200, 300, 500, 800, 1000]
print(f'{"arch":30s} ' + ' '.join(f'{s:>10d}' for s in steps))
for k in ('v6', 'gdn', 'tf'):
    d = D[k]
    if not d:
        continue
    row = []
    for s in steps:
        vals = []
        for sd in d:
            for x in d[sd].get('snapshots', []):
                if x['step'] == s:
                    vals.append(x['loss_mean_recent'])
        row.append(statistics.median(vals) if vals else float('nan'))
    print(f'{names[k]:30s} ' + ' '.join(f'{v:10.0f}' for v in row))

print('\n=== 收敛判定（末 200 step 降幅）===')
for k in ('v6', 'gdn', 'tf'):
    d = D[k]
    if not d:
        continue
    for sd in sorted(d):
        snaps = d[sd].get('snapshots', [])
        bys = {x['step']: x['loss_mean_recent'] for x in snaps}
        ks = sorted(bys)
        if len(ks) >= 5:
            prev, last = bys[ks[-5]], bys[ks[-1]]
            drop = (prev - last) / prev * 100
            print(f'  {names[k]:28s} s{sd}: {prev:.0f} → {last:.0f}  降 {drop:5.2f}%  '
                  f'{"已到平台" if drop < 1 else "仍在下降"}')

print('\n=== 参数量与墙钟（形态不同，速度不可直接比）===')
for k in ('v6', 'gdn', 'tf'):
    d = D[k]
    if not d:
        continue
    s0 = d[sorted(d)[0]]
    print(f'  {names[k]:28s} params={s0["student_params"]:12,}  '
          f'wall 中位={statistics.median([d[s]["wall_sec"] for s in d]):7.0f}s  peak={s0["peak_gib"]}GiB')
