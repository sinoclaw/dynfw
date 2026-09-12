"""修正版因果性探针 —— 严格对齐 full/cut 的 fast-weight 状态, 查 v6/v7 是否真因果。
上一版探针发现非因果, 但 block 里明明有 causal mask (tril diagonal=-1),
矛盾→要么探针bug, 要么某处泄漏。本版逐位置、逐块模拟, 隔离泄漏点。
关键: 对位置 t, full(全序列)与 cut(截断到t)前向时, 每个 block 的 mem 必须一致。
"""
import torch
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM

def probe(arch, vocab=256, D=128, nh=16, n_layer=2, mlp_mult=64, W=256, T=20, seed=0):
    torch.manual_seed(seed)
    if arch == 'v6':
        m = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)
    else:
        m = BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W, K=8)
    m.eval()
    x = torch.randint(0, vocab, (1, T))
    with torch.no_grad():
        lg_full, _ = m.forward(x, None)   # [1,T,V]
    worst = 0; worst_t = -1
    for t in range(1, T):
        xc = x[:, :t+1]
        with torch.no_grad():
            lg_cut, _ = m.forward(xc, None)
        d = (lg_full[0, t] - lg_cut[0, t]).abs().max().item()
        if d > worst:
            worst = d; worst_t = t
    print(f"[{arch}] W={W} T={T} 最大 diff={worst:.6f} @t={worst_t} "
          f"-> {'严格因果✓' if worst < 1e-4 else f'非因果✗ diff={worst:.4f}'}")
    return worst

if __name__ == '__main__':
    print("=== v6 修正因果探针 (W=256 大块) ===")
    probe('v6', W=256)
    print("=== v6 小块 W=64 (块内因果更明显) ===")
    probe('v6', W=64)
    print("=== v7 DLA (W=256) ===")
    probe('v7', W=256)
    print("=== v7 DLA (W=64) ===")
    probe('v7', W=64)
