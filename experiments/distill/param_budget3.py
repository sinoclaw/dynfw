"""扫三架构逼近同一【总参数预算】(含词表, 部署口径)的配置。

词表税 v = 2*vocab*D (未tie)。总参 = 词表税 + 结构参数。
目标: 找三架构 (level D / n_layer / mlp_mult) 使【总参数量级一致】。
"""
import torch
vocab = 151936

def pp(D, nh, nl, mmp, arch):
    if arch == 'la_cycle':
        from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
        return BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=nl, steps=2, mlp_mult=mmp).np()
    if arch == 'bdh':
        from dynfw.models.bdh_qwen import BDHQwen
        return BDHQwen(D=D, n_layer=nl, nh=nh, mlp_mult=mmp, vocab=vocab, dropout=0.0).np()
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=nh, n_layer=nl, vocab=vocab, maxT=512).np()

print("=== 三架构总参数(含词表)逼近同一预算 ===")
print(f"{'arch':10s} {'D':>5s} {'nh':>4s} {'L':>3s} {'mmp':>5s} {'total(M)':>9s} {'结构(M)':>8s} {'词表(M)':>8s}")
print("-"*68)
# 找几组配置, 看总参数趋近
# 思路: D 增大 → 词表税 2*vocab*D 增大 + 结构增大
for D, nh, mmp in [(128, 16, 64), (256, 16, 32), (384, 16, 16), (256, 16, 64)]:
    vt = 2*vocab*D/1e6
    for nl in [2, 4]:
        for arch in ['la_cycle', 'bdh', 'tf']:
            tot = pp(D, nh, nl, mmp, arch)
            print(f"{arch:10s} {D:>5d} {nh:>4d} {nl:>3d} {mmp:>5d} {tot/1e6:>9.1f} {(tot-vt*1e6)/1e6:>8.1f} {vt:>8.1f}")
        print()
