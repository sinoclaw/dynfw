"""同 D=512 下 v7 DLA vs 标准 TF 推理显存随 T 对比 —— 直答"是否比 TF 更吃显存"。
DLA L=4 (357M, 无KV-cache固定状态) vs TF L=8 (181M, 带KV-cache)。
军规：报增速（每翻倍T的显存增量），不只看单点。
"""
import torch, gc
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
from dynfw.models.transformer import TF_sdpa
VOCAB = 151936

dla = BDHBlockDLACycleLM(D=512, nh=16, vocab=VOCAB, n_layer=4, steps=1, mlp_mult=64, W=256, K=8).cuda().eval()
tf = TF_sdpa(D=512, nh=16, n_layer=8, vocab=VOCAB, maxT=16384).cuda().eval()
print(f"DLA 357M vs TF {sum(p.numel() for p in tf.parameters())/1e6:.0f}M (D=512)")

print(f"\n{'T':>7} | {'DLA GB':>8} | {'TF GB':>8} | DLA/TF")
prev_dla = prev_tf = None
for T in [256, 512, 1024, 2048, 4096, 8192, 16384]:
    row = []
    for name, m in [('dla', dla), ('tf', tf)]:
        x = torch.randint(0, VOCAB, (1, T)).cuda()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad(): m(x)
            row.append(torch.cuda.max_memory_allocated() / 1e9)
        except RuntimeError as e:
            row.append(999.0)
        torch.cuda.empty_cache(); gc.collect(); del x
    d, t = row
    ratio = d / t if t < 900 else 999
    # 增速
    gd = f"+{(d-prev_dla):.2f}" if prev_dla else "-"
    gt = f"+{(t-prev_tf):.2f}" if prev_tf else "-"
    print(f"{T:>7} | {d:>8.2f} | {t:>8.2f} | {ratio:>5.2f}x  (DLA{gd} TF{gt})")
    prev_dla, prev_tf = d, t
