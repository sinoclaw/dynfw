"""v7 DLA 单卡4090 显存/参数边界探测：扫 D×L×mlp_mult，测 forward+backward 峰值显存。
军规：新候选先量化单卡边界再立项，不拍脑袋报规模。"""
import torch
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM

VOCAB = 151936
MHZ = 1e6

def probe(D, L, nh, mlp_mult, W=256, K=8, batch=1, T=512):
    """跑一次 forward+backward，返回 参数(M) + 峰值显存(GB)。"""
    m = BDHBlockDLACycleLM(D=D, nh=nh, vocab=VOCAB, n_layer=L, steps=1,
                           mlp_mult=mlp_mult, W=W, K=K).cuda().train()
    params = m.np() / MHZ
    x = torch.randint(0, VOCAB, (batch, T)).cuda()
    t = torch.randint(0, VOCAB, (batch, T)).cuda()
    torch.cuda.reset_peak_memory_stats()
    lg, loss = m(x, t)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache()
    del m, x, t, lg, loss
    torch.cuda.empty_cache()
    return params, peak

print(f"vocab={VOCAB} 单卡4090(24GB) v7 DLA 显存/参数边界 (batch=1, T=512)")
print(f"{'D':>5} {'L':>3} {'nh':>4} {'mlp':>5} | {'总参数M':>8} {'结构M':>7} {'词表税M':>8} | {'峰值GB':>8} | 判定")
print("-" * 78)
for D, L, nh, mlp in [
    (128, 2, 16, 64),   # 当前配置(基线)
    (128, 4, 16, 64),   # 加层
    (128, 8, 16, 64),   # 深度
    (256, 2, 16, 64),   # 加宽
    (256, 4, 16, 64),
    (384, 2, 16, 64),
    (384, 4, 16, 64),
    (512, 2, 16, 64),
    (256, 4, 16, 32),   # 降mlp
    (512, 2, 16, 32),
]:
    try:
        p, pk = probe(D, L, nh, mlp)
        struct = (p - 2*VOCAB*D/MHZ)
        ver = "OOM" if pk > 23.5 else ("OK" if pk < 22 else "临界")
        print(f"{D:>5} {L:>3} {nh:>4} {mlp:>5} | {p:>8.1f} {struct:>7.1f} {2*VOCAB*D/MHZ:>8.1f} | {pk:>8.2f} | {ver}")
    except RuntimeError as e:
        print(f"{D:>5} {L:>3} {nh:>4} {mlp:>5} | OOM/CRASH: {str(e)[:40]}")
    torch.cuda.empty_cache()
