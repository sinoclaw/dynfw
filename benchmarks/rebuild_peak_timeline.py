"""用时间线重建「峰值时刻同时在世的分配」，按调用栈聚合。

方法：
  遍历 alloc/free_completed 事件（按 time_us 排序），维护在世集合，
  记录「在世字节总量」达到最大的那一刻，输出那一刻的构成。
这才是真正的峰值构成（不是累计分配量）。
"""
import pickle, collections, os, sys

GU = 2 ** 30


def rebuild(path, tag):
    snap = pickle.load(open(path, 'rb'))
    ev = snap['device_traces'][0]
    ev = sorted(ev, key=lambda e: e['time_us'])

    alive = {}          # addr -> event
    cur = 0
    peak = 0
    peak_alive = None
    peak_time = None

    for e in ev:
        a = e['action']
        if a == 'alloc':
            alive[e['addr']] = e
            cur += e['size']
            if cur > peak:
                peak = cur
                peak_alive = list(alive.values())
                peak_time = e['time_us']
        elif a == 'free_completed':
            old = alive.pop(e['addr'], None)
            if old:
                cur -= old['size']

    print("=" * 104)
    print(f"[{tag}] 重建峰值 = {peak/GU:.4f} GiB   （事件数 {len(ev)}，峰值时刻 {peak_time}）")
    print("=" * 104)

    def key_of(frames):
        for fr in frames:
            fn = fr.get('filename') or ''
            if 'dynfw' in fn:
                return f"{os.path.basename(fn)}:{fr.get('line')} {fr.get('name')}"
        for fr in frames:
            fn = fr.get('filename') or ''
            return f"{os.path.basename(fn)}:{fr.get('line')} {fr.get('name')}"
        return '(unknown)'

    agg = collections.defaultdict(lambda: [0, 0])
    for e in peak_alive or []:
        k = key_of(e.get('frames') or [])
        agg[k][0] += e['size']
        agg[k][1] += 1

    print(f"  峰值时刻在世 {len(peak_alive or [])} 块，按调用栈聚合：")
    print()
    for k, (byt, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:16]:
        print(f"  {byt/GU:8.4f} GiB  ×{n:<3d}  {k[:92]}")
    print()
    return peak


p0 = rebuild('/tmp/memsnap_base.pickle', 'base')
p1 = rebuild('/tmp/memsnap_ckpt.pickle', 'ckpt（grad_ckpt=True）')
print(f"对比：{p0/GU:.4f} GiB → {p1/GU:.4f} GiB   "
      f"（降 {100*(p0-p1)/p0:.1f}%）")
