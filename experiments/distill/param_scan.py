#!/usr/bin/env python3
"""实测各架构在同配置下的结构参数(不含词表 embed/head), 找结构≈40M 的配置, 并验证各架构结构对齐。"""
import torch, sys
sys.path.insert(0, '/data/dynfw')

def scan(D, nh, nl, mlp, W, arch, vocab=151936, K=8):
    from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    from dynfw.models.bdh_qwen import BDHQwen
    from dynfw.models.transformer import TF_sdpa
    from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
    if arch == 'v6':
        m = BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=nl, steps=1, mlp_mult=mlp, W=W)
    elif arch == 'la_cycle':
        m = BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=nl, steps=1, mlp_mult=mlp)
    elif arch == 'bdh':
        m = BDHQwen(D=D, n_layer=nl, nh=nh, mlp_mult=mlp, vocab=vocab, dropout=0.0)
    elif arch == 'tf':
        m = TF_sdpa(D=D, nh=nh, n_layer=nl, vocab=vocab, maxT=W)
    elif arch == 'dla':
        m = BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=nl, steps=1, mlp_mult=mlp, W=W, K=K)
    total = m.np()
    struct = total - 2*vocab*D   # 结构 = 总 - 词表 tax (embed+head)
    return total, struct

for (D, nh, nl, mlp) in [(192,16,3,64),(224,16,4,64),(256,16,3,64),(256,16,4,48),(256,16,3,72),(224,16,4,72)]:
    print(f"--- D={D} nh={nh} nl={nl} mlp={mlp} ---")
    for a in ['v6','la_cycle','bdh','tf','dla']:
        try:
            tot, st = scan(D, nh, nl, mlp, 256, a)
            print(f"  {a:10s} 总={tot/1e6:6.2f}M 结构={st/1e6:6.2f}M")
        except Exception as e:
            print(f"  {a:10s} ERR {e}")
