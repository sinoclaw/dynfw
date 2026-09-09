"""复杂度实测 v2: 用更大 T 范围 + 多次测量取均值, 让 O(T) vs O(T²) 差异显现。"""
import torch, time, math
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM

def probe(mk, label, Ts):
    m = mk().cuda().eval()
    times = []
    for T in Ts:
        x = torch.randint(0, 151936, (1, T)).cuda()
        with torch.no_grad(): m(x)  # 预热 + 触发内存分配
        torch.cuda.synchronize()
        # 多次取均值
        t0 = time.time(); n = 20
        for _ in range(n):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize()
        times.append((time.time() - t0) / n)
    print(f"{label}:")
    for T, t in zip(Ts, times):
        print(f"   T={T:5d}  step={t*1000:6.2f}ms")
    ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
    import numpy as np
    # 线性回归斜率: log2(T) vs log2(time)
    logT = np.log2(Ts); logTt = np.log2(times)
    sl = np.polyfit(logT, logTt, 1)[0]
    print(f"   log-log 斜率={sl:.2f}  (O(T)线≈1.0, O(T²)≈2.0)  翻倍比={[round(r,2) for r in ratios]}")
    del m; torch.cuda.empty_cache()

Ts = [1024, 2048, 4096, 8192]
print("=== 复杂度实测 v2 (log-log 斜率, O(T)≈1.0 / O(T²)≈2.0) ===")
probe(lambda: BDHBlockFWCycleLM(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=512), "v6 FW(fast-weight)", Ts)
# v5 在 T=8192 会 OOM (之前测 T=8192 是 19.5GB), 试到 4096
probe(lambda: BDHBlockCycleLM(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64), "v5 la_cycle(O(T²)整段)", Ts[:3])
