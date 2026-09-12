"""v5 vs v6 vs gdn 的 8-seed 差异是否统计显著（Welch t-test，无 scipy 则手算）。"""
import glob
import json
import math
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


v5 = vals([f'{R}/retest_fusedfw_la_cycle_s*/**/*.json', f'{R}/v5anchor_s*/**/*.json'])
v6 = vals([f'{R}/readmode_raw_s*/**/*.json'])
gdn = vals([f'{R}/gdnread_raw_s*/**/*.json'])


def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, se, t, df


def p_two_sided(t, df):
    """用 Student-t 的生存函数近似（数值积分）。"""
    x = df / (df + t * t)
    # 正则化不完全 beta 的连分数近似
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    a, b = df / 2.0, 0.5
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1.0 - x) / b
    return ib  # = 双尾 p 值（I_x(df/2, 1/2)）


pairs = [('v5 la_cycle', v5, 'v6 raw', v6), ('v5 la_cycle', v5, 'gdn raw', gdn),
         ('v6 raw', v6, 'gdn raw', gdn)]
print(f"{'比较':34s} {'Δ均值':>8s} {'SE':>7s} {'t':>7s} {'df':>6s} {'p(双尾)':>9s} 判定")
print('-' * 92)
for na, a, nb, b in pairs:
    d, se, t, df = welch(a, b)
    p = p_two_sided(abs(t), df)
    verdict = '显著 (p<0.05)' if p < 0.05 else ('边缘 (p<0.1)' if p < 0.1 else '**不显著**')
    print(f'{na+" vs "+nb:34s} {d:8.2f} {se:7.3f} {t:7.3f} {df:6.1f} {p:9.3f} {verdict}')

print()
print('判读：若 v5 vs v6 不显著 ⇒ 「O(T) 压缩架构在能力上与 O(T^2) 精确架构统计等价」成立。')
print('      若 v6 vs gdn 不显著 ⇒ 门控(v6.6)无收益，v6.6 可退役。')
