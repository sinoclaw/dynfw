"""验证 v6/v7 因果性 —— 位置 t 的输出是否只依赖 <=t 的输入（不看未来）。
方法: 对每个位置 t, 比较 全序列前向 的 logits[t] 与 截断到t 的 logits[t].
若因果: 二者应完全一致 (因为 t 处不应看到 t 之后的内容).
若非因果(用了未来): 二者不同, 且 diff 随 t 增大(未来间隔变大).
军规: 这是评测框架适用性的硬前提, 不通过则 loglikelihood 基准(MMLU/ARC)会失真.
"""
import torch
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM

def causal_probe(arch, vocab=256, D=128, nh=16, n_layer=2, mlp_mult=64, T=16, seed=0):
    torch.manual_seed(seed)
    if arch == 'v6':
        m = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256)
    else:
        m = BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256, K=8)
    m.eval()
    x = torch.randint(0, vocab, (1, T))
    # 全序列前向
    with torch.no_grad():
        logits_full, _ = m.forward(x, targets=None)  # [1,T,V]
    print(f"[{arch}] 全序列 logits shape: {tuple(logits_full.shape)}")
    # 逐位置: 截断到 t 前向 对比
    max_diff = 0.0
    infos = []
    for t in range(1, T):
        x_t = x[:, :t+1]  # 截断到 t (含t)
        with torch.no_grad():
            logits_cut, _ = m.forward(x_t, targets=None)
        # 比较位置 t 的 logits
        l_full = logits_full[0, t].cpu()
        l_cut = logits_cut[0, t].cpu()
        diff = (l_full - l_cut).abs().max().item()
        max_diff = max(max_diff, diff)
        infos.append((t, diff))
    print(f"[{arch}] 逐位置  位置t: 全序列 vs 截断t 的 logits diff:")
    for t, d in infos:
        print(f"    t={t:2d}: diff={d:.6f}  {'因果✓' if d < 1e-5 else '非因果✗'}")
    print(f"[{arch}] RESULT max_diff={max_diff:.6f}  -> {'严格因果(可上loglikelihood基准)' if max_diff < 1e-5 else '非因果(用了未来, loglikelihood会失真)'}")
    return max_diff

if __name__ == '__main__':
    print("=== v6 因果性探针 ===")
    causal_probe('v6')
    print("\n=== v7 因果性探针 ===")
    causal_probe('v7')
