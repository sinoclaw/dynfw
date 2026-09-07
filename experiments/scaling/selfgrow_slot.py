"""Self-growing-slot CPU 探针：固定 D，训练中动态涨 slot 数 N，能否比一开始给满 N 更有效？

背景（GPT 审计，仅本地存档）：FusedFW 的"自生长"应该走"固定 D + 动态涨 slot 数 N"，
而非"宽度 D 扩宽 + 权重继承"（后者已证 No-Go）。核心问题：
  **固定 D 的 FusedFW，训练中自主 spawn 新 slot，能否在"同最终参数 + 同预算"下
  比"一开始就给满 slot 数"获得更好/不降的 loss？**

设计（判据先行，一次一组变量，P0 seed-先建模型）：
  D 固定 = 96，k=16，use_ffn=True，N(槽数) 是唯一被动的变量。
  - Fixed-96 : N=96 从零，训 B_total=300（"一开始给满"基线）
  - Fixed-32 : N=32 从零，训 300（reference：小容量是否容量受限/饱和）
  - GrowA     : N=32 训 B1=150 → grow 到 96（新槽 fresh-init，不继承）→ 全模型续训 B2=150
                （GPT"新槽独立学 + 旧能力参数不动"的轻量版：不冻结，全靠新槽承接新容量）
  - GrowB     : 同 GrowA 但 grow 后"冻结旧槽行梯度"（仅新槽真正学）→ 更贴近 GPT"旧参数完全不动"

  同数据 / 同 val / 同 token budget(300 步) / 同最终 N=96（同最终参数）/ 3 seed。

判定：
  GrowA 或 GrowB 的最终 val ≤ Fixed-96（同最终参数同预算下能力不降/更好）→ GPT 的"capacity-growth
  优于 fixed"假说成立；若均 > Fixed-96 → 自增长 slot 在 CPU 上仍不优于"一给满"，像宽度 No-Go 一样
  证伪（诚实记录）。另看 Fixed-32 是否饱和（residual 高 = 容量压力确实存在）。
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time
from dynfw.data import load_data, get_batch

torch.set_num_threads(8)
D = 96; K = 16; VOCAB = 256; LR = 3e-4
B_TOTAL = 300; B1 = 150; B2 = 150
SEEDS = [0, 1, 2]
N0 = 32; NFULL = 96


def make_val(block=256, nseq=64, seed=123):
    data = load_data(); val = data[int(0.9*len(data)):]
    rng = np.random.RandomState(seed); ix = rng.randint(0, len(val)-block, (nseq,))
    x = torch.stack([torch.from_numpy(val[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(val[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x, y
VAL = make_val()


class FusedFW_Grow(nn.Module):
    """Fixed-D FusedFW，enc 是 [N,D] 参数，支持 grow_to(N_new)（新槽 fresh-init，不继承）。"""
    def __init__(self, D=D, N=N0, k=K, vocab=VOCAB, use_ffn=True):
        super().__init__(); self.D=D; self.k=k; self.vocab=vocab; self.use_ffn=use_ffn
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, vocab)
        self.enc_W = nn.Parameter(torch.randn(N, D) * 0.02)   # [N,D]，N 可 grow（新增槽 fresh init）
        self.dec_W = nn.Parameter(torch.randn(D, N) * 0.02)   # [D,N]（原实现 dec 未用于计算，保参数对账）
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    @property
    def N(self): return self.enc_W.size(0)

    def forward(self, x, t=None):
        B, T = x.size(); D = self.D; N = self.N
        et = self.e(x)                        # B,T,D
        lat = et @ self.enc_W.T               # B,T,N
        topv, topi = torch.topk(lat, self.k, dim=-1)
        act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # B,T,N 稀疏
        ln_et = self.ln(et)
        rho = torch.einsum('btn,btd->bnd', act, ln_et)   # B,N,D
        mem = torch.einsum('btn,bnd->btd', act, rho)     # B,T,D
        h = et + mem
        h = self.ln(h); h = self.out(h)
        if self.use_ffn: h = h + self.ffn(self.ln(h))
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def grow_to(self, N_new):
        oldN = self.N
        assert N_new > oldN, "只能 grow 不能 shrink"
        with torch.no_grad():
            ne = torch.randn(N_new, self.D) * 0.02; ne[:oldN] = self.enc_W
            self.enc_W = nn.Parameter(ne)
            nd = torch.randn(self.D, N_new) * 0.02; nd[:, :oldN] = self.dec_W
            self.dec_W = nn.Parameter(nd)
        return oldN  # 返回旧 N，供冻结判据用

    def np(self): return sum(p.numel() for p in self.parameters())


def val_now(m):
    m.eval()
    with torch.no_grad():
        xv, yv = VAL; _, vl = m(xv, yv)
    return float(vl.item())


def train(m, iters, seed, freeze_rows_after=None, log_every=30, curves=None):
    """训练；freeze_rows_after=(start_row) 则在每次 backward 后把 >=start_row 的行梯度清零(冻结旧槽)。
    返回最终 val，并把 val 曲线写进 curves dict。"""
    torch.manual_seed(seed); np.random.seed(seed)
    data = load_data(); rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for i in range(iters):
        m.train(); x, y = get_batch(data, 256, 8, rng); _, loss = m(x, y)
        opt.zero_grad(); loss.backward()
        if freeze_rows_after is not None:
            # 冻结旧槽行（< oldN）：把旧行梯度清零，只有新槽行更新
            with torch.no_grad():
                m.enc_W.grad[:freeze_rows_after] = 0
        opt.step()
        if log_every and (i + 1) % log_every == 0 and curves is not None:
            curves[(i + 1)] = val_now(m)
    return val_now(m)


def run():
    print(f"=== Self-growing-slot CPU 探针 (固定 D={D}, 涨 N: {N0}->{NFULL}, k={K}) ===", flush=True)
    print(f"  同数据/同val/同token budget({B_TOTAL}步)/同最终N={NFULL}(同最终参数)/{len(SEEDS)}seed\n", flush=True)
    res = {k: {'vals': [], 'curves': []} for k in ['Fixed-96', 'Fixed-32', 'GrowA(全参续训)', 'GrowB(冻结旧槽)']}
    t0 = time.time()
    for sd in SEEDS:
        # Fixed-96
        torch.manual_seed(sd); np.random.seed(sd)
        m = FusedFW_Grow(D=D, N=NFULL); c = {}
        vf = train(m, B_TOTAL, sd, curves=c); res['Fixed-96']['vals'].append(vf); res['Fixed-96']['curves'].append(c)
        # Fixed-32
        torch.manual_seed(sd); np.random.seed(sd)
        m = FusedFW_Grow(D=D, N=N0); c = {}
        v32 = train(m, B_TOTAL, sd, curves=c); res['Fixed-32']['vals'].append(v32); res['Fixed-32']['curves'].append(c)
        # GrowA (全参续训): N0 训 B1 -> grow -> 全模型训 B2
        torch.manual_seed(sd); np.random.seed(sd)
        m = FusedFW_Grow(D=D, N=N0); c = {}
        train(m, B1, sd, curves=c)
        oldN = m.grow_to(NFULL)
        va = train(m, B2, sd, curves=c); res['GrowA(全参续训)']['vals'].append(va); res['GrowA(全参续训)']['curves'].append(c)
        # GrowB (冻结旧槽): 同上，但 B2 阶段冻结旧槽行梯度
        torch.manual_seed(sd); np.random.seed(sd)
        m = FusedFW_Grow(D=D, N=N0); c = {}
        train(m, B1, sd, curves=c)
        oldN = m.grow_to(NFULL)
        vb = train(m, B2, sd, freeze_rows_after=oldN, curves=c); res['GrowB(冻结旧槽)']['vals'].append(vb); res['GrowB(冻结旧槽)']['curves'].append(c)
        print(f"  seed{sd}: Fixed-96={vf:.3f} Fixed-32={v32:.3f} GrowA={va:.3f} GrowB={vb:.3f}", flush=True)

    # 各配置真实参数量（最终 N 下）
    res['Fixed-96']['params'] = FusedFW_Grow(D=D, N=NFULL).np()
    res['Fixed-32']['params'] = FusedFW_Grow(D=D, N=N0).np()
    res['GrowA(全参续训)']['params'] = FusedFW_Grow(D=D, N=NFULL).np()
    res['GrowB(冻结旧槽)']['params'] = FusedFW_Grow(D=D, N=NFULL).np()

    print("\n=== 汇总 (mean±std, {} seed) ===".format(len(SEEDS)), flush=True)
    for k, r in res.items():
        v = np.array(r['vals'])
        print(f"  {k:<18} {v.mean():.3f} ± {v.std():.3f}   params≈{r.get('params', '?'):,}", flush=True)
    def mm(k): return float(np.mean(res[k]['vals']))
    print(f"\n→ 判定: GrowA(vs Fixed-96, 同最终参数+同预算): {mm('GrowA(全参续训)')-mm('Fixed-96'):+.3f}", flush=True)
    print(f"→ 判定: GrowB(vs Fixed-96): {mm('GrowB(冻结旧槽)')-mm('Fixed-96'):+.3f}", flush=True)
    print(f"→ 参考: Fixed-32(N0 直接训满) = {mm('Fixed-32'):.3f}  (< 更大N 说明 N={N0} 确实容量受限/有容量压力)", flush=True)
    import json
    json.dump({'D': D, 'N0': N0, 'NFULL': NFULL, 'k': K, 'B_total': B_TOTAL, 'seeds': SEEDS,
               'results': {k: {'vals': r['vals'], 'curves': r['curves']} for k, r in res.items()}},
              open('results/v0.0.1/selfgrow_slot.json', 'w'), indent=2)
    print(f"\nDONE ({time.time()-t0:.0f}s) saved results/v0.0.1/selfgrow_slot.json", flush=True)


if __name__ == '__main__':
    run()
