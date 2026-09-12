"""收口实验汇总：v6(opt5) vs TF，T=8192，1000 step，3 seed。
判据（跑前锁死）：1000step loss 中位 + Welch t-test；末 200 step 降幅 <1% 视为收敛。
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
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        out[j['seed']] = j
    return out


v6, tf = load('v6'), load('tf')
print(f'v6 seeds: {sorted(v6)}   tf seeds: {sorted(tf)}')

# ── 曲线 ──
print('\n=== 各 seed 的 loss 曲线（step: loss_mean_recent）===')
for tag, d in (('v6', v6), ('tf', tf)):
    for s in sorted(d):
        snaps = d[s].get('snapshots', [])
        pts = ' '.join(f'{x["step"]}:{x["loss_mean_recent"]:.0f}' for x in snaps)
        print(f'  {tag} s{s}: {pts}')
        print(f'         total_steps={d[s].get("total_steps")}  wall={d[s]["wall_sec"]:.0f}s  '
              f'final={d[s]["final_loss"]:.1f}')

# ── 各预算点中位对比 ──
print('\n=== 预算点对比（中位 loss）===')
print(f'{"step":>6s} {"v6":>12s} {"tf":>12s} {"v6/tf":>8s} {"胜者":>6s} {"Welch p":>10s}')
for target in (100, 200, 300, 500, 800, 1000):
    def val(d):
        vs = []
        for s in sorted(d):
            for x in d[s].get('snapshots', []):
                if x['step'] == target:
                    vs.append(x['loss_mean_recent'])
        return vs
    a, b = val(v6), val(tf)
    if len(a) < 1 or len(b) < 1:
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    # Welch t
    if len(a) > 1 and len(b) > 1:
        va, vb = statistics.variance(a), statistics.variance(b)
        se = math.sqrt(va / len(a) + vb / len(b))
        tt = (statistics.mean(a) - statistics.mean(b)) / se if se > 0 else 0.0
        df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)) if se > 0 else 0
        # 粗略 p（双尾，用 t 分布近似）
        try:
            from scipy import stats as sps
            p = 2 * (1 - sps.t.cdf(abs(tt), df))
            ps = f'{p:.4f}'
        except Exception:
            ps = f't={tt:.2f}'
    else:
        ps = 'n=1'
    print(f'{target:6d} {ma:12.1f} {mb:12.1f} {ma/mb:8.4f} {"v6" if ma < mb else "tf":>6s} {ps:>10s}')

# ── 收敛判定 ──
print('\n=== 收敛判定（末 200 step 降幅）===')
for tag, d in (('v6', v6), ('tf', tf)):
    for s in sorted(d):
        snaps = d[s].get('snapshots', [])
        bys = {x['step']: x['loss_mean_recent'] for x in snaps}
        ks = sorted(bys)
        if len(ks) >= 5:
            last, prev = bys[ks[-1]], bys[ks[-5]]
            drop = (prev - last) / prev * 100
            print(f'  {tag} s{s}: step{ks[-5]}={prev:.0f} → step{ks[-1]}={last:.0f}  '
                  f'降幅 {drop:.2f}%  {"已到平台" if drop < 1.0 else "仍在下降"}')

# ── 参数量/速度 ──
print('\n=== 参数量与墙钟 ===')
for tag, d in (('v6', v6), ('tf', tf)):
    s0 = d[sorted(d)[0]]
    print(f'  {tag}: params={s0["student_params"]:,}  wall 中位='
          f'{statistics.median([d[s]["wall_sec"] for s in d]):.0f}s  '
          f'peak={s0["peak_gib"]}GiB')
print('\n注：v6 用 opt5_raw（交付形态），tf 用 SDPA 融合核 —— 同优化等级。')
