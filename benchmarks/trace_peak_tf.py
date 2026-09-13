"""TF 的峰值溯源（与 v6.7 同配置、同方法、同代理 head）。

配置：D=128, nh=16, n_layer=2, T=8192, batch=1, 代理 head=4096
方法：_record_memory_history → 时间线重建 → 按调用栈聚合峰值时刻在世分配
"""
import sys, pickle, collections, os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T, B = 8192, 1
D, NH, NLAYER, VOCAB = 128, 16, 2, 151936
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True

from dynfw.models.transformer import TF_sdpa

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)

torch.manual_seed(0)
m = TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
nparam = sum(p.numel() for p in m.parameters())
print(f"TF 参数量 = {nparam:,}")


def one():
    o = m.forward_hidden(x)
    lg = o.view(B * T, D) @ head.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return loss.item()


for _ in range(2):
    one()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

torch.cuda.memory._record_memory_history(max_entries=200000, context='all')
lv = one()
torch.cuda.synchronize()
path = '/tmp/memsnap_tf.pickle'
torch.cuda.memory._dump_snapshot(path)
torch.cuda.memory._record_memory_history(enabled=None)
pk = torch.cuda.max_memory_allocated() / GiB
print(f"[TF] torch 报的 peak={pk:.4f} GiB  loss={lv:.6f}")


def rebuild(path, tag):
    snap = pickle.load(open(path, 'rb'))
    ev = sorted(snap['device_traces'][0], key=lambda e: e['time_us'])
    alive, cur, peak, peak_alive = {}, 0, 0, None
    for e in ev:
        if e['action'] == 'alloc':
            alive[e['addr']] = e; cur += e['size']
            if cur > peak:
                peak, peak_alive = cur, list(alive.values())
        elif e['action'] == 'free_completed':
            old = alive.pop(e['addr'], None)
            if old: cur -= old['size']

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
        agg[k][0] += e['size']; agg[k][1] += 1

    print()
    print("=" * 104)
    print(f"[{tag}] 重建峰值 = {peak/GiB:.4f} GiB （在世 {len(peak_alive or [])} 块）")
    print("=" * 104)
    for k, (byt, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:16]:
        print(f"  {byt/GiB:8.4f} GiB  ×{n:<3d}  {k[:92]}")
    return peak


rebuild(path, 'TF')
