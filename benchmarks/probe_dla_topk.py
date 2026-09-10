"""DLA 状态槽 + MoBA 式槽选择 —— 跑前锁死的三个探针 (v2, 口径修正版)。

判据(跑前锁定, 跑后如实对照):
  P1 正确性 : v8 read_mode='sum' 与 v7 原版【同 seed 同权重同输入】→ logits maxdiff < 1e-6
              (证 v8 的 sum 分支真等价原版, 无实现 bug) + 参数量 Δ=0 (ksum 是状态非参数)
  P2 尺度/恒等式 : 逐块(used=k)检验代数恒等式  α≡1/k  ⟹  retr_uni ≡ retr_sum / k
                  并报各 mode 逐块 L2 —— 看"选择"是否悄悄改了信号尺度(红线)
  P3 复杂度 : (a) 整模型前向 wall-clock vs T, log2-log2 斜率 (O(T)≈1.0 / O(T²)≈2.0)
              (b) 【只测 attn 模块】隔离读侧成本 —— 关键问题:
                  sum 读 = O(w·N·D) ; 选择性读 = O(w·u·N·D) → 选择是【更贵】还是更省?
     v2 修正: 上一版把 used 当常数做聚合比值(错, used 逐块 1..k); 且测了 agg+retr 总输出
              (块内注意稀释了差异)。本版逐块、且用 capture_retr 只抓检索部分。
"""
import sys, time, json, math
sys.path.insert(0, '/data/dynfw')
import torch

torch.set_num_threads(8)
OUT = {}

from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
from dynfw.models.fused_fw_dla_topk_cycle import BDHBlockSlotCycleLM, DLASlotAttn, Config

# ---------------- P1 正确性 ----------------
print("=" * 74)
print("P1 正确性: v8 read_mode='sum' vs v7 原版 (同 seed/权重/输入)")
print("=" * 74)
CFG = dict(D=128, nh=16, mlp_mult=64, vocab=4096, n_layer=1, steps=1, W=128, K=4)
torch.manual_seed(0)
v7 = BDHBlockDLACycleLM(**CFG).eval()
torch.manual_seed(0)
v8 = BDHBlockSlotCycleLM(read_mode='sum', **CFG).eval()
miss, unexp = v8.load_state_dict(v7.state_dict(), strict=False)
p7 = sum(p.numel() for p in v7.parameters())
p8 = sum(p.numel() for p in v8.parameters())
torch.manual_seed(1)
x = torch.randint(0, 1000, (1, 512))
with torch.no_grad():
    d = (v7.forward_logits(x) - v8.forward_logits(x)).abs().max().item()
print(f"  state_dict missing={list(miss)} unexpected={list(unexp)}")
print(f"  参数: v7={p7:,}  v8={p8:,}  Δ={p8-p7}   (须=0: ksum 是状态不是参数)")
print(f"  ▶ logits maxdiff = {d:.3e}   {'PASS ✓' if d < 1e-6 else 'FAIL ✗'}")
OUT.update(P1_maxdiff=d, P1_params=p7)


# ---------------- P2 逐块尺度 + 恒等式 ----------------
def capture(mode, force_uniform=False, topk=2, K=16, W=128, T=1024, temp=1.0):
    torch.manual_seed(0)
    cfg = Config(1, 128, 16, 64, 4096)
    a = DLASlotAttn(cfg, K=K, read_mode=mode, topk=topk, temp=temp).eval()
    a.force_uniform = force_uniform
    a.capture_retr = True
    q = torch.randn(1, 16, T, 512) * 0.1
    v = torch.randn(1, 1, T, 128)
    with torch.no_grad():
        _, cache = a(q, q, v, None, W=W)
    return a.last_retr, cache[4]


print()
print("=" * 74)
print("P2 尺度: 逐块(used=k)的检索 L2 + 代数恒等式 α≡1/k ⟹ retr_uni ≡ retr_sum/k")
print("=" * 74)
caps = {m: capture(m)[0] for m in ['sum', 'softmax', 'softmaxK', 'topk']}
uni, used = capture('softmax', force_uniform=True)

# 逐块恒等式: 第 i 个捕获块 (0-based) 对应 used = i+1
errs = []
for i, (a, b) in enumerate(zip(caps['sum'], uni)):
    k = i + 1
    errs.append(((b * k - a).abs().max() / (a.abs().max() + 1e-12)).item())
print(f"  逐块恒等式 max 相对误差 = {max(errs):.3e}   {'PASS ✓' if max(errs) < 1e-4 else 'FAIL ✗'}")
OUT['P2_identity_maxrelerr'] = max(errs)

print()
print("  块k  |  sum L2   | softmax   ratio | softmaxK  ratio | topk(k=2)  ratio")
for i in range(len(caps['sum'])):
    k = i + 1
    s = caps['sum'][i].norm(dim=-1).mean().item()
    row = f"  k={k:2d} | {s:9.4f}"
    for m in ['softmax', 'softmaxK', 'topk']:
        val = caps[m][i].norm(dim=-1).mean().item()
        row += f" | {val:9.4f} {val/s:.2f}x"
    print(row)
print(f"\n  ▶ 理论: softmax ≈ sum/k, softmaxK ≈ sum, topk(k=2) ≈ sum·(2/k)  ← 归一化是否动尺度")
OUT['P2_scales'] = {m: [round(c.norm(dim=-1).mean().item(), 4) for c in caps[m]] for m in caps}

# ---------------- P3 复杂度 ----------------
print()
print("=" * 74)
print("P3 复杂度")
print("=" * 74)
Ts = [512, 1024, 2048, 4096]


def slope_of(xs, ys):
    lx = [math.log2(x) for x in xs]; ly = [math.log2(y) for y in ys]
    n = len(lx); mx = sum(lx) / n; my = sum(ly) / n
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


print("  (a) 整模型前向:")
for mode in ['sum', 'softmaxK', 'topk']:
    ts = []
    for T in Ts:
        torch.manual_seed(0)
        m = BDHBlockSlotCycleLM(D=128, nh=16, mlp_mult=64, vocab=4096, n_layer=1,
                                steps=1, W=512, K=8, read_mode=mode, topk=2).eval()
        xx = torch.randint(0, 1000, (1, T))
        best = 1e9
        with torch.no_grad():
            for _ in range(3):
                t0 = time.perf_counter(); m.forward_logits(xx); best = min(best, time.perf_counter() - t0)
        ts.append(best)
    print(f"     {mode:10s} times={[round(t,4) for t in ts]}  slope={slope_of(Ts, ts):.3f}")

print("  (b) 【只测 attn 模块】隔离读侧成本 (固定 W=512, K=16, used 已满):")
for mode in ['sum', 'softmaxK', 'topk']:
    ts = []
    for T in Ts:
        torch.manual_seed(0)
        a = DLASlotAttn(Config(1, 128, 16, 64, 4096), K=16, read_mode=mode, topk=2).eval()
        q = torch.randn(1, 16, T, 512) * 0.1
        v = torch.randn(1, 1, T, 128)
        best = 1e9
        with torch.no_grad():
            for _ in range(3):
                t0 = time.perf_counter(); a(q, q, v, None, W=512); best = min(best, time.perf_counter() - t0)
        ts.append(best)
    print(f"     {mode:10s} times={[round(t,4) for t in ts]}  slope={slope_of(Ts, ts):.3f}")
print("     ⚠️ 关键: sum 读 = O(w·N·D); 选择性读 = O(w·u·N·D) → 选择本身【更贵 u 倍】")
OUT['P3_Ts'] = Ts
print()
print(json.dumps(OUT, ensure_ascii=False))
