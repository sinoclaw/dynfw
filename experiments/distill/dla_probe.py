"""DLA状态槽范数+复杂度实测: v7(DLA) vs v6(单块) vs v6.6(门控) 对比。"""
import torch, math, time
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM as V7
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM as V6
import numpy as np

print("=== 范数对比 (Σ‖M‖: V7状态槽 vs V6单块) D=128,nh=16,L=1,W=64 ===")
print(f"{'T':>6} | {'V6单块':>12} | {'V7状态槽':>12}")
for T in [256, 512, 1024, 2048, 4096, 8192]:
    m7 = V7(D=128, nh=16, vocab=151936, n_layer=1, steps=1, mlp_mult=64, W=64).cuda().eval()
    m6 = V6(D=128, nh=16, vocab=151936, n_layer=1, steps=1, mlp_mult=64, W=64).cuda().eval()
    x = torch.randint(0, 151936, (1, T)).cuda()
    with torch.no_grad():
        h = m6.e(x).unsqueeze(1); h = m6.ln(h); mem6 = None
        for b in m6.blocks: h, mem6 = b(h, mem6)
        h7 = m7.e(x).unsqueeze(1); h7 = m7.ln(h7); mem7 = None
        for b in m7.blocks: h7, mem7 = b(h7, mem7)
    n6 = mem6.float().norm().item()
    S, I, n, used = mem7
    n7 = S[:, :, :used].float().norm().item()
    print(f"{T:>6} | {n6:>12.1f} | {n7:>12.1f}")
    del m6, m7; torch.cuda.empty_cache()

print("\n=== 复杂度实测 (log-log斜率, O(T)≈1.0/O(T²)≈2.0) V7 DLA ===")
def probe(mk, label):
    m = mk().cuda().eval()
    times = []; Ts = [1024, 2048, 4096, 8192]
    for T in Ts:
        x = torch.randint(0, 151936, (1, T)).cuda()
        with torch.no_grad(): m(x)
        torch.cuda.synchronize(); t0 = time.time(); n = 15
        for _ in range(n):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize(); times.append((time.time()-t0)/n)
    sl = np.polyfit(np.log2(Ts), np.log2(times), 1)[0]
    print(f"{label}: 斜率={sl:.2f} ([{round(t*1000,1)}ms x {len(Ts)}]) {'O(T)' if sl<1.45 else 'O(T²)'}")
    del m; torch.cuda.empty_cache()

probe(lambda: BDHBlockDLACycleLM(D=128, nh=16, vocab=151936, n_layer=1, steps=1, mlp_mult=64, W=64), "V7 DLA状态槽")
probe(lambda: BDHBlockFWCycleLM(D=128, nh=16, vocab=151936, n_layer=1, steps=1, mlp_mult=64, W=64), "V6 单块")
