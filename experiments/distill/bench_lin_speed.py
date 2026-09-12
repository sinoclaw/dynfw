"""快速测 FusedFW_lin 的 prefill 翻倍比 + decode，验证是否真 O(T) 线性。"""
import torch, time
from dynfw.models.fused_fw_lin import FusedFWLin
from dynfw.models.fused_fw_full import FusedFWFull
from dynfw.models.transformer import TF_sdpa

VOCAB = 151936
T_LIST = [256, 512, 1024, 2048]
N_REPEAT = 5

def mk_lin():
    return FusedFWLin(D=128, nh=4, dk=32, vocab=VOCAB, n_layer=2, use_ffn=True)

def time_prefill(model, T):
    model.eval()
    x = torch.randint(0, VOCAB, (2, T), device='cuda')
    with torch.no_grad(): model(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(): model(x)
        torch.cuda.synchronize()
        ts.append(time.time()-t0)
    return min(ts)

def time_decode(model, n=64):
    model.eval()
    x = torch.randint(0, VOCAB, (1, 32), device='cuda')
    with torch.no_grad():
        for _ in range(3): model(x)
        torch.cuda.synchronize(); t0 = time.time()
        cur = x
        for _ in range(n):
            with torch.no_grad():
                lg, _ = model(cur)
            cur = torch.cat([cur, lg[:, -1:].argmax(dim=-1)], dim=1)
        torch.cuda.synchronize()
        tot = time.time()-t0
    return tot/n

if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    lin = mk_lin().to('cuda')
    print(f"FusedFW_lin params: {lin.np()}")
    print("\n=== PREFILL 扫T (batch=2) ===")
    row = [time_prefill(lin, T) for T in T_LIST]
    print(f"T={T_LIST}")
    print(f"ms={[f'{v*1000:.1f}' for v in row]}")
    for i in range(1, len(T_LIST)):
        print(f"  翻倍比 T={T_LIST[i-1]}->{T_LIST[i]}: {row[i]/row[i-1]:.2f}")
    ms = time_decode(lin)*1000
    print(f"\nDECODE 每token: {ms:.2f} ms")
    print(f"\n【判据】翻倍比 ~1.0=O(T)线性, ~2.0=O(T²)")
