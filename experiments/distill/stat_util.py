"""纯 stdlib 的 Welch t-test（不依赖 scipy）。

为什么自带：GPU venv 无 scipy，pip 安装超时；统计判定不该被依赖阻塞。
实现 = Numerical Recipes 的 incomplete beta 连分数法，精度足够（|误差| < 1e-10）。
"""
import math


def _betacf(a, b, x, itmax=200, eps=3e-16, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """正则化不完全 Beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """双尾 p 值 P(|T| > |t|)。"""
    if df <= 0:
        return float('nan')
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def welch(a, b):
    """Welch t 检验 -> (t, df, p)。"""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, float(n1 + n2 - 2), 1.0
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, df, t_two_sided_p(t, df)


if __name__ == '__main__':
    # 自检：与已知临界值对照
    checks = [(4.604, 4, 0.01), (8.610, 4, 0.001), (2.776, 4, 0.05), (12.706, 1, 0.05)]
    print('自检（t, df, 期望p, 计算p）:')
    for t, df, want in checks:
        got = t_two_sided_p(t, df)
        print(f'  t={t:7.3f} df={df} 期望={want:.4f} 计算={got:.6f} {"OK" if abs(got-want) < 2e-3 else "MISMATCH"}')
