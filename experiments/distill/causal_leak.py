"""隔离 v6 非因果泄漏源 —— 逐 block 单独前向, 逐块比对 full/cut 的 mem 与 out。
定位: 是 causal mask 失效? 还是 new_mem 跨块检索把未来带进来? 还是位置差?
"""
import torch
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM

def probe_single_arch(D=128, nh=16, n_layer=2, mlp_mult=64, W=256, T=20, vocab=256, seed=0):
    torch.manual_seed(seed)
    m = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)
    m.eval()
    x = torch.randint(0, vocab, (1, T))
    # 手动跑一个 block 看因果性
    blk = m.blocks[0]
    h_full = m.e(x).unsqueeze(1)  # [B,1,T,D]
    h_full = m.ln(h_full)
    # 全序列
    out_full, mem_full = blk(h_full, None)
    # 截断到 t
    worst = 0
    for t in range(1, T):
        h_cut = h_full[:, :, :t+1].contiguous()
        out_cut, mem_cut = blk(h_cut, None)
        d = (out_full[0, :, t] - out_cut[0, :, t]).abs().max().item()
        if d > worst: worst = d
    print(f"[block0] 块本身前向: full vs cut@t 最大 diff={worst:.6f} {'因果✓' if worst<1e-4 else '非因果✗'}")
    # 再看 block 内部: 检查 causal mask 是否真的只用 tril
    import inspect
    src = inspect.getsource(blk.forward)
    print("   block.forward 源码关键行:")
    for line in src.split('\n'):
        if 'tril' in line or 'masked' in line or 'causal' in line:
            print("     ", line.strip())

probe_single_arch()
