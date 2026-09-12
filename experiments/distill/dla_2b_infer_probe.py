"""2B 级 v7 DLA 推理显存扫描 —— 找上 24GB 的 2B 配置，测 decode 短序与长上下文。
军规：推理口径用 decode（逐 token 生成），非 prefill 整个序列；分短序/长上下文报。"""
import torch, gc
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
VOCAB = 151936

def probe(D, L, mlp, T):
    m = BDHBlockDLACycleLM(D=D, nh=16, vocab=VOCAB, n_layer=L, steps=1,
                           mlp_mult=mlp, W=256, K=8).cuda().eval()
    params = m.np() / 1e6
    x = torch.randint(0, VOCAB, (1, T)).cuda()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            m(x)
        pk = torch.cuda.max_memory_allocated() / 1e9
        return params, pk
    finally:
        torch.cuda.empty_cache(); gc.collect()
        del m, x

print("=== 2B 级 v7 DLA 推理显存 (decode短序 T=8 / 长上下文 T=4096, batch=1) ===")
print(f"{'D':>6} {'L':>3} {'mlp':>4} | {'paramsM':>8} | {'T=8':>8} {'T=4096':>8} | 判定")
for D, L, mlp in [(1024, 8, 32), (1280, 8, 32), (1536, 8, 32), (1536, 6, 32), (1152, 10, 32)]:
    try:
        p1, pk1 = probe(D, L, mlp, 8)
        try:
            _, pk2 = probe(D, L, mlp, 4096)
        except RuntimeError:
            pk2 = 999.0
        worst = max(pk1, pk2)
        ver = "OOM" if worst > 23.5 else ("OK" if worst < 22 else "临界")
        print(f"{D:>6} {L:>3} {mlp:>4} | {p1:>8.0f} | {pk1:>8.2f} {pk2:>8.2f} | {ver}")
    except RuntimeError as e:
        print(f"{D:>6} {L:>3} {mlp:>4} | CRASH {str(e)[:30]}")
    torch.cuda.empty_cache(); gc.collect()
