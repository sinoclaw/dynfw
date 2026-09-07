"""Growth-inheritance 实验：从小模型 warm-start 扩宽到更大模型，能否复用已学权重？

背景（爸爸问）：DynFW v0.0.1 是固定结构，100K → 更大要重头训练。本实验验证
"能否从小模型的已学权重扩宽到更大模型"——即"增量长大 + 复用"是否有戏。

设计（判据先行，一次一组变量）：
  路径 A  fresh-big：D=96 从零随机初始化，总预算 = B_total
  路径 B  grow：     D=64 先训 B_1，扩宽到 D=96（零填充继承已学权重），再训 B_2
                     要求 B_1 + B_2 ≈ B_total（同总训练预算）
  路径 C  fresh-big-same-B2：D=96 从零随机初始化只训 B_2（隔离"大模型短训"效应）

判据（跑前锁死）：
  - 若 B（warmstart 复用）最终 val 明显优于 A（重头训同预算）→ 增量长大复用有戏；
  - 若 B ≈ A 或 B 更差 → warmstart 不能省训练，静态架构"每档重训"是常态，复用无显著收益。

横向一致性：N = 4*D（保持不变），k=16，use_ffn=True，同数据/同 iters/同 budget/同 seed。
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np
from dynfw.models.fused_fw import FusedFW
from dynfw.data import load_data, get_batch

torch.set_num_threads(8)
D_SMALL = 64; D_BIG = 96       # N=4*D
K = 16
B_TOTAL = 300; B_1 = 150; B_2 = 150
SEEDS = [0, 1, 2]
LR = 3e-4


def make_val(block=256, nseq=64, seed=123):
    data = load_data(); val = data[int(0.9*len(data)):]
    rng = np.random.RandomState(seed); ix = rng.randint(0, len(val)-block, (nseq,))
    x = torch.stack([torch.from_numpy(val[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(val[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x, y

VAL = make_val()


def train(model, iters, seed, bl=256, bt=8, lr=LR, valset=VAL):
    """seed 先设再建模型（真独立初始化）。返回 (模型, 最终val)。"""
    torch.manual_seed(seed); np.random.seed(seed)
    data = load_data(); rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(iters):
        model.train(); x, y = get_batch(data, bl, bt, rng); _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        xv, yv = valset; _, vl = model(xv, yv)
    return float(vl.item())


def grow(small, D_big):
    """零填充扩宽：small(FusedFW D0,N0) -> D_big。新维度清零（中性起跑），旧块精确继承。"""
    D0 = small.D; N0 = small.N; N_big = 4 * D_big
    big = FusedFW(D=D_big, N=N_big, k=K, use_ffn=True)
    with torch.no_grad():
        # e: Embedding[vocab,D_big]；旧 D 列复制，新列清零
        big.e.weight.zero_(); big.e.weight[:, :D0] = small.e.weight
        # ln: 旧维复制，新维 weight=1 bias=0
        big.ln.weight.zero_(); big.ln.weight[:D0] = small.ln.weight; big.ln.weight[D0:] = 1.0
        big.ln.bias.zero_(); big.ln.bias[:D0] = small.ln.bias
        # out: [D_big,D_big]；旧块复制，新行/列清零
        big.out.weight.zero_(); big.out.weight[:D0, :D0] = small.out.weight
        # head: [vocab,D_big]；旧 D 列复制，新列清零
        big.head.weight.zero_(); big.head.weight[:, :D0] = small.head.weight
        for i in range(small.n_layer):
            big.encs[i].weight.zero_(); big.encs[i].weight[:N0, :D0] = small.encs[i].weight
            big.decs[i].weight.zero_(); big.decs[i].weight[:D0, :N0] = small.decs[i].weight
        if small.use_ffn:
            big.ffn[0].weight.zero_(); big.ffn[0].weight[:4*D0, :D0] = small.ffn[0].weight
            big.ffn[2].weight.zero_(); big.ffn[2].weight[:D0, :4*D0] = small.ffn[2].weight
    return big


def run():
    print(f"=== Growth-inheritance: warmstart 扩宽能否复用已学权重 ===\n"
          f"  D_small={D_SMALL}(N={4*D_SMALL}) D_big={D_BIG}(N={4*D_BIG}) k={K} "
          f"B_total={B_TOTAL} (B1={B_1}+B2={B_2})\n", flush=True)
    rows = {p: [] for p in ['A_fresh_300', 'B_grow_150_150', 'C_fresh_150']}
    for sd in SEEDS:
        # A: fresh big D=96, B_total（P0: 先 seed 再建模型，seed 真控初始化）
        torch.manual_seed(sd); np.random.seed(sd)
        mkA = FusedFW(D=D_BIG, N=4*D_BIG, k=K, use_ffn=True)
        va = train(mkA, B_TOTAL, sd)
        # B: small D=64 train B1 -> grow to D_big -> train B2（同 sd，seed-先建）
        torch.manual_seed(sd); np.random.seed(sd)
        small = FusedFW(D=D_SMALL, N=4*D_SMALL, k=K, use_ffn=True)
        train(small, B_1, sd)
        bigb = grow(small, D_BIG)
        vb = train(bigb, B_2, sd)
        # C: fresh big D=96 only B2
        torch.manual_seed(sd); np.random.seed(sd)
        mkC = FusedFW(D=D_BIG, N=4*D_BIG, k=K, use_ffn=True)
        vc = train(mkC, B_2, sd)
        rows['A_fresh_300'].append(va); rows['B_grow_150_150'].append(vb); rows['C_fresh_150'].append(vc)
        print(f"  seed{sd}: A(fresh300)={va:.3f}  B(grow150+150)={vb:.3f}  C(fresh150)={vc:.3f}", flush=True)

    print("\n=== 结果（mean±std）===", flush=True)
    for p, vs in rows.items():
        print(f"  {p:<18} {np.mean(vs):.3f} ± {np.std(vs):.3f}", flush=True)
    ma = np.mean(rows['A_fresh_300']); mb = np.mean(rows['B_grow_150_150'])
    print(f"\n→ B(grow) vs A(fresh同预算): {mb - ma:+.3f}  ({'warmstart 优/能省' if mb < ma else '重头训更优/复用无增益'})", flush=True)
    import json
    json.dump({k: v for k, v in rows.items()}, open('results/v0.0.1/growth_inheritance.json', 'w'), indent=2)
    print("\nDONE saved results/v0.0.1/growth_inheritance.json", flush=True)


if __name__ == '__main__':
    run()
