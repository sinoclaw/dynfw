"""Growth-inheritance 机理诊断：B(grow) 为何劣于 A(fresh)？

把 B−A 分解为两个可测分量：
  ① 预算机会成本：B 在最终规模 D=96 只训 B_2 iters（A 训满 B_TOTAL）；
  ② 机制惩罚：grow 瞬间（零填充 + LayerNorm 重归一化到 96 维）对继承权重的扰动。

⚠️ P0 seed：模型必须在 manual_seed 之后创建（factory 先 seed 再建），seed 才真控初始化。
方法（每 seed 一条 val 曲线，3 seed）：
  - 小模型  D64 训 B_1 → 每 30 iters 记 val，得 V_small_end
  - grow → D96，记 V_grow0（增长瞬间质量；V_grow0 − V_small_end = grow 机制扰动代价）
  - 大模型 D96 续训 B_2 → 记 val，得 B_end
  - 全新 D96 从零训 B_TOTAL → 记 30/150/300，得 C(=fresh@150) 与 A(=fresh@300)
"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
import torch, numpy as np
from dynfw.models.fused_fw import FusedFW
from dynfw.data import load_data, get_batch
from growth_inheritance import D_SMALL, D_BIG, K, B_1, B_2, LR, make_val, grow

torch.set_num_threads(8)
VAL = make_val()
BT = 8


def make_fw(D):
    return FusedFW(D=D, N=4*D, k=K, use_ffn=True)


def train_curve(model, iters, batch_seed, eval_every=30):
    """训练 model（调用方已 seed-先建模型）。batch 采样按 batch_seed 推进（忠实原实验）。"""
    data = load_data(); rng = np.random.RandomState(batch_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = {}
    for i in range(iters):
        model.train()
        x, y = get_batch(data, 256, 8, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = VAL; _, vl = model(xv, yv)
            curve[i + 1] = float(vl.item())
    model.eval()
    with torch.no_grad():
        xv, yv = VAL; _, vl = model(xv, yv)
    return float(vl.item()), curve


def val_now(model):
    model.eval()
    with torch.no_grad():
        xv, yv = VAL; _, vl = model(xv, yv)
    return float(vl.item())


def run_seed(sd):
    root = sd * 1000
    # --- 路径 B: small D64 → grow → D96 续训（一条 seed 流，不中断）---
    torch.manual_seed(root + 1); np.random.seed(root + 1)
    small = make_fw(D_SMALL)
    vs, csmall = train_curve(small, B_1, batch_seed=root + 1)
    bigb = grow(small, D_BIG)
    vg0 = val_now(bigb)
    vbig, cbig = train_curve(bigb, B_2, batch_seed=root + 1)
    # --- 路径 A/C: 全新 D96 从零训 B_TOTAL（独立 seed）---
    torch.manual_seed(root + 2); np.random.seed(root + 2)
    fresh = make_fw(D_BIG)
    vend, cfresh = train_curve(fresh, B_TOTAL := 300, batch_seed=root + 2)
    c150 = cfresh.get(150, float('nan')); a300 = cfresh.get(300, vend)
    return dict(small_end=vs, grow0=vg0, B_end=vbig, C=c150, A=a300,
                small_curve=csmall, big_curve=cbig, fresh_curve=cfresh)


def run():
    print(f"=== Growth-inheritance 机理诊断 (D64→D96, P0 seed 修) — 分解预算 vs 机制 ===\n", flush=True)
    print(f"  D_small={D_SMALL}(N={4*D_SMALL}) D_big={D_BIG}(N={4*D_BIG}) k={K} "
          f"B_total=300 (B1={B_1}+B2={B_2}) lr={LR}\n", flush=True)
    SEEDS = [0, 1, 2]
    out = []
    for sd in SEEDS:
        r = run_seed(sd); out.append(r)
        print(f"  seed{sd}: small_end={r['small_end']:.3f} grow0={r['grow0']:.3f} "
              f"(jump={r['grow0']-r['small_end']:+.3f})  B_end={r['B_end']:.3f} "
              f"C@150={r['C']:.3f} A@300={r['A']:.3f}", flush=True)

    def m(k): return float(np.mean([r[k] for r in out]))
    def s(k): return float(np.std([r[k] for r in out]))
    print("\n=== 汇总 (mean±std, {} seed) ===".format(len(SEEDS)), flush=True)
    for k in ['small_end', 'grow0', 'B_end', 'C', 'A']:
        print(f"  {k:<10} {m(k):.3f} ± {s(k):.3f}", flush=True)
    jump = m('grow0') - m('small_end'); warm = m('B_end') - m('C'); gapBA = m('B_end') - m('A')
    print(f"\n→ grow 机制扰动 (grow0−small_end): {jump:+.3f}  (正=增长瞬间丢的)", flush=True)
    print(f"→ warmstart 净增益 @150 (B−C): {warm:+.3f}", flush=True)
    print(f"→ B−A (最终同预算): {gapBA:+.3f}", flush=True)
    print("→ 判定: 若 warmstart(=B−C)<0 且 B−A>0,")
    print("   则 warmstart 是常数小偏移、不随训练复利; B 主因=预算机会成本(final 规模只训 B_2)。", flush=True)
    import json
    json.dump(dict(seeds=SEEDS, summary={k: [m(k), s(k)] for k in ['small_end','grow0','B_end','C','A']},
                   per_seed=out), open('results/v0.0.1/growth_inheritance_diag.json','w'), indent=2)
    print("\nDONE saved results/v0.0.1/growth_inheritance_diag.json", flush=True)


if __name__ == '__main__':
    run()
