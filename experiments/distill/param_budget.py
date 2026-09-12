"""参数预算估算: 扫 la_cycle/bdh_qwen/TF_sdpa 三架构在固定 D 下的结构参数, 找同预算对齐配置。"""
import torch

def p_la_cycle(D, nh, n_layer, mlp_mult, vocab, steps=2):
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    m = BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=steps, mlp_mult=mlp_mult)
    return m.np()

def p_bdh(D, nh, n_layer, mlp_mult, vocab):
    from dynfw.models.bdh_qwen import BDHQwen
    m = BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab, dropout=0.0)
    return m.np()

def p_tf(D, nh, n_layer, vocab, block):
    from dynfw.models.transformer import TF_sdpa
    m = TF_sdpa(D=D, nh=nh, n_layer=n_layer, vocab=vocab, maxT=block)
    return m.np()

print(f"{'架构':10s} {'D':>5s} {'nh':>4s} {'L':>3s} {'struct(M)':>10s} {'struct(M)去词表':>16s}")
print("-"*60)
# 点1: D=128, mlp_mult=128, 各层数
for nl in [2, 4]:
    for D, mmp, nh in [(128, 128, 16)]:
        la = p_la_cycle(D, nh, nl, mmp, 151936)
        bd = p_bdh(D, nh, nl, mmp, 151936)
        tf = p_tf(D, nh, nl, 151936, 512)
        # 去词表(embed+lm_head): vocab*D*2(la/bdh 不tie) / tf不tie同
        vocab_tax = 151936 * D * 2
        print(f"{'la_cycle':10s} {D:>5d} {nh:>4d} {nl:>3d} {la/1e6:>10.1f} {(la-vocab_tax)/1e6:>16.2f}")
        print(f"{'bdh':10s}     {D:>5d} {nh:>4d} {nl:>3d} {bd/1e6:>10.1f} {(bd-vocab_tax)/1e6:>16.2f}")
        print(f"{'tf':10s}      {D:>5d} {nh:>4d} {nl:>3d} {tf/1e6:>10.1f} {(tf-vocab_tax)/1e6:>16.2f}")
        print()
