"""汇总长 T(8192) 实验：J3 训练等价性 + opt5 vs TF 双壁（能力+速度）+ 旧批参考。

用法: python summarize_lt8192.py
"""
import glob
import json
import os
import statistics

R = '/data/dynfw/results'


def load(prefix):
    out = {}
    for d in sorted(glob.glob(os.path.join(R, prefix + '*'))):
        if not os.path.isdir(d) or 'smoke' in d:
            continue
        f = os.path.join(d, 'distill_result.json')
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        out.setdefault(j['arch'], {})[j['seed']] = j
    return out


def stat(vs):
    if not vs:
        return None
    return (len(vs), statistics.median(vs), statistics.mean(vs),
            statistics.stdev(vs) if len(vs) > 1 else 0.0)


def show(tag, d):
    print(f'--- {tag} ---')
    for arch in sorted(d):
        rows = d[arch]
        losses = [rows[s]['final_loss'] for s in sorted(rows)]
        walls = [rows[s]['wall_sec'] for s in sorted(rows)]
        s = stat(losses)
        print(f'  {arch:22s} n={s[0]}  中位loss={s[1]:12.2f} 均值={s[2]:12.2f} std={s[3]:8.2f}  '
              f'中位wall={statistics.median(walls):7.1f}s')
        print(f'{"":26s}{[round(v, 1) for v in losses]}')
    print()


opt5 = load('lt8192opt5_fw_')
base_new = load('lt8192base_fw_')
tf = load('lt8192opt5_tf_')
old = load('lt8192_')

show('新批：opt5 形态（交付形态）', opt5)
show('新批：基线形态（J3 对照）', base_new)
show('新批：TF', tf)
show('旧批：全部基线形态（未优化，仅参考）', old)

print('=' * 78)
print('【J3】训练等价性：opt5 vs 基线（同 seed，全 20 epoch）')
for seed in sorted(set(opt5.get('fusedfw_fw_cycle', {})) & set(base_new.get('fusedfw_fw_cycle', {}))):
    a = opt5['fusedfw_fw_cycle'][seed]
    b = base_new['fusedfw_fw_cycle'][seed]
    d = a['final_loss'] - b['final_loss']
    ok = 'PASS' if abs(d) < 0.02 else 'FAIL'
    print(f'  seed={seed}  opt5={a["final_loss"]:12.4f}  base={b["final_loss"]:12.4f}  '
          f'Δ={d:+.6f}  (判据 |Δ|<0.02)  {ok}')
    print(f'           wall: opt5={a["wall_sec"]:7.1f}s  base={b["wall_sec"]:7.1f}s  '
          f'提速 {b["wall_sec"]/a["wall_sec"]:.2f}x')

print()
print('【双壁】opt5 vs TF（同优化等级）')
fw = opt5.get('fusedfw_fw_cycle', {})
tfs = tf.get('tf', {})
if fw and tfs:
    lf = [fw[s]['final_loss'] for s in sorted(fw)]
    lt = [tfs[s]['final_loss'] for s in sorted(tfs)]
    wf = [fw[s]['wall_sec'] for s in sorted(fw)]
    wt = [tfs[s]['wall_sec'] for s in sorted(tfs)]
    print(f'  opt5(v6): n={len(lf)} loss 中位={statistics.median(lf):12.2f}  wall 中位={statistics.median(wf):7.1f}s')
    print(f'  TF      : n={len(lt)} loss 中位={statistics.median(lt):12.2f}  wall 中位={statistics.median(wt):7.1f}s')
    print(f'  ⇒ 能力比(TF/opt5，>1 表示 TF 更好): {statistics.median(lt)/statistics.median(lf):.4f}')
    print(f'  ⇒ 速度比(opt5 wall / TF wall，<1 表示我们更快): {statistics.median(wf)/statistics.median(wt):.2f}')
    # Welch t
    import math
    if len(lf) > 1 and len(lt) > 1:
        va, vb = statistics.variance(lf), statistics.variance(lt)
        se = math.sqrt(va / len(lf) + vb / len(lt))
        t = (statistics.mean(lf) - statistics.mean(lt)) / se
        df = (va / len(lf) + vb / len(lt)) ** 2 / (
            (va / len(lf)) ** 2 / (len(lf) - 1) + (vb / len(lt)) ** 2 / (len(lt) - 1))
        print(f'  Welch t-test(opt5 vs TF): Δ均值={statistics.mean(lf)-statistics.mean(lt):+.2f} '
              f'SE={se:.2f} t={t:+.3f} df={df:.1f}')
else:
    print('  (数据不足)')
