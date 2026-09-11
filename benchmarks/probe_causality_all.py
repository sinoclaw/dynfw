"""通用因果性探针：长度依赖法（比扰动法更便宜、无需先猜泄漏在哪）。

原理：因果模型里，位置 POS 的输出只依赖 token[0..POS]，
      所以喂 seg[:POS+1] 与 seg[:POS+2]，POS 处 logits 必须逐位相同(maxdiff == 0)。

用法: PYTHONPATH=/data/dynfw python benchmarks/probe_causality_all.py
"""
import sys, traceback
import torch, torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# (名称, 构造 lambda)  —— 全部用 n_layer=2/3 以暴露跨 block 泄漏
def build_all(D=64, nh=4, mlp_mult=16, W=64, vocab=256, n_layer=3):
    from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
    from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
    from dynfw.models.fused_fw_dla_topk_cycle import BDHBlockSlotCycleLM
    from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM
    from dynfw.models.fused_fw_rawfw_cycle import BDHBlockRawFWCycleLM
    from dynfw.models.fused_fw_vla_cycle import BDHBlockVLACycleLM
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    from dynfw.models.transformer import TF_sdpa
    from dynfw.models.bdh_qwen import BDHQwen
    from dynfw.models.bdh_rawfw_qwen import BDHRawFWQwen

    out = [
        ('transformer(TF_sdpa)',  lambda: TF_sdpa(D=D, nh=nh, vocab=vocab, n_layer=n_layer)),
        ('v6 fused_fw_fw_cycle',  lambda: BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)),
        ('la_cycle',              lambda: BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, use_ffn=True, n_layer=n_layer, steps=1, mlp_mult=mlp_mult)),
        ('v7 dla_cycle',          lambda: BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W, K=8)),
        ('v8 dla_topk_cycle',     lambda: BDHBlockSlotCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W, K=16, topk=4)),
        ('v6.6 gdn_cycle',        lambda: BDHBlockGDNCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)),
        ('rawfw_cycle',           lambda: BDHBlockRawFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)),
        ('vla_cycle',             lambda: BDHBlockVLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=W)),
        ('bdh_qwen(O(T^2) 整段)',  lambda: BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab)),
        ('bdh_rawfw_qwen',        lambda: BDHRawFWQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab, W=W)),
    ]
    return out


def probe(m, T, POS, seed=0):
    """返回位置 POS 处的 logits。"""
    torch.manual_seed(seed)
    seg = torch.randint(0, 200, (T + 2,), device=DEV)
    x1 = seg[:POS + 1].unsqueeze(0)
    x2 = seg[:POS + 2].unsqueeze(0)
    with torch.no_grad():
        m.eval()
        l1 = m(x1)[0][0, POS, :].float()
        l2 = m(x2)[0][0, POS, :].float()
    return (l1 - l2).abs().max().item()


def main():
    print(f"device={DEV}")
    print(f"{'架构':<28}{'maxdiff(T=POS+1 vs POS+2)':>28}{'判定':>10}")
    print('-' * 68)
    results = {}
    for name, ctor in build_all():
        try:
            m = ctor().to(DEV)
            diffs = [probe(m, T=40, POS=30), probe(m, T=80, POS=60), probe(m, T=64, POS=20)]
            d = max(diffs)
            verdict = 'CAUSAL OK' if d < 1e-6 else 'LEAK'
            results[name] = (d, verdict)
            print(f"{name:<28}{d:>28.6e}{verdict:>10}")
        except Exception as e:
            results[name] = (float('nan'), 'ERROR')
            print(f"{name:<28}{'--':>28}{'ERROR':>10}  {type(e).__name__}: {e}")
    print('-' * 68)
    bad = [k for k, (d, v) in results.items() if v != 'CAUSAL OK']
    print(f"不合格: {bad if bad else '无 —— 全部因果 ✓'}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
