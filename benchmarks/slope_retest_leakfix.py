"""复杂度斜率复验（泄漏修复后）：各架构 log-log 斜率是否仍为真 O(T)。
口径与 W=64 公平矩阵一致（W=64 → T/W 个 chunk，跨块记忆真被触发）。
判据：O(T) ≈ 1.0，O(T²) ≈ 2.0；须扫足够大的 T 范围（小 T 被 GPU 延迟主导会低估）。
"""
import time

import numpy as np
import torch

from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM
from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
from dynfw.models.transformer import TF_sdpa

DEV = 'cuda'
KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)


def probe(name, mk, Ts, reps=10):
    try:
        m = mk().to(DEV).eval()
    except Exception as e:
        print(f'{name}: 构建失败 {type(e).__name__}: {e}', flush=True)
        return
    times, used = [], []
    for T in Ts:
        x = torch.randint(0, 151936, (1, T), device=DEV)
        try:
            with torch.no_grad():
                m(x)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'  {name}: T={T} 前向失败 → 截断({str(e)[:70]})', flush=True)
            break
        t0 = time.time()
        for _ in range(reps):
            with torch.no_grad():
                m(x)
        torch.cuda.synchronize()
        times.append((time.time() - t0) / reps)
        used.append(T)
        del x
    if len(times) >= 2:
        sl = np.polyfit(np.log2(used), np.log2(times), 1)[0]
        kind = 'O(T) 线性 ✅' if sl < 1.3 else ('O(T²) 平方 ❌' if sl > 1.6 else '介于两者 ⚠️')
        print(f'{name:30s} 斜率={sl:.2f}  {kind}', flush=True)
        for T, t in zip(used, times):
            print(f'      T={T:6d}  {t * 1000:8.2f} ms   (chunk数={T // 64})', flush=True)
    del m
    torch.cuda.empty_cache()


Ts = [1024, 2048, 4096, 8192]
print('=== 复杂度斜率复验（log-log；O(T)≈1.0 / O(T²)≈2.0）===', flush=True)
probe('v6 fw_cycle (W=64)', lambda: BDHBlockFWCycleLM(**KW), Ts)
probe('v6.6 gdn_cycle (W=64)', lambda: BDHBlockGDNCycleLM(**KW), Ts)
probe('v5 la_cycle (预期 O(T²))', lambda: BDHBlockCycleLM(**{k: v for k, v in KW.items() if k != 'W'}), Ts)
probe('TF_sdpa (预期 O(T²))', lambda: TF_sdpa(D=128, nh=16, n_layer=2, vocab=151936, maxT=8192), Ts)
print('=== 复验结束 ===', flush=True)