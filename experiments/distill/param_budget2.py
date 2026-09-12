"""反推对齐: 扫每个架构的层数/mlp_mult, 使去词表结构参数逼近同一目标预算。"""
import torch
vocab = 151936
D = 128; nh = 16

def p_la_cycle(n_layer, mlp_mult):
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    return BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=2, mlp_mult=mlp_mult).np()

def p_bdh(n_layer, mlp_mult):
    from dynfw.models.bdh_qwen import BDHQwen
    return BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab, dropout=0.0).np()

def p_tf(n_layer, block=512):
    from dynfw.models.transformer import TF_sdpa
    return TF_sdpa(D=D, nh=nh, n_layer=n_layer, vocab=vocab, maxT=block).np()

vocab_tax = vocab * D * 2  # embed + lm_head (都不tie)
print("目标: 同结构参数预算对轰 (去词表)")
def fmt(m):
    return f"{m/1e6:.2f}M (结构{(m-vocab_tax)/1e6:.2f}M)"

print("\n--- 扫描 la_cycle / bdh (mlp_mult=32, 让小N结构合理) ---")
for mmp in [32, 64]:
    for nl in [2, 4, 6, 8]:
        la = p_la_cycle(nl, mmp); bd = p_bdh(nl, mmp)
        print(f"  mmp={mmp} L={nl}: la_cycle={fmt(la)}  bdh={fmt(bd)}")

print("\n--- 扫描 tf (L 增大逼近结构预算) ---")
for nl in [4, 6, 8, 12, 16]:
    tf = p_tf(nl)
    print(f"  tf L={nl}: {fmt(tf)}")
