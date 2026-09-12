"""三方推理速度基准：FusedFWFull / BDH / TF
测 prefill（扫 T 看 O(T) vs O(T²)）+ decode（逐token）。
判据：wall-clock 翻倍比 ~2.0=O(T²)、~0.5-1.0=O(T)。
"""
import torch, time, os
from dynfw.models.fused_fw_full import FusedFWFull
from dynfw.models.bdh_qwen import BDHQwen
from dynfw.models.transformer import TF_sdpa

VOCAB = 151936
T_LIST = [256, 512, 1024, 2048]
N_REPEAT = 5
DECODE_STEPS = 64


def make(name, D=128, n_layer=2, nh=4, N=512):
    if name == 'fusedfw_full':
        return FusedFWFull(D=D, N=N, k=16, nh=nh, mlp_mult=32, vocab=VOCAB, n_layer=n_layer)
    if name == 'bdh':
        return BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=32, vocab=VOCAB, dropout=0.0)
    if name == 'tf':
        return TF_sdpa(D=D, nh=nh, n_layer=n_layer, vocab=VOCAB, maxT=2048)
    raise ValueError(name)


def time_prefill(model, name, T):
    model.eval()
    x = torch.randint(0, VOCAB, (2, T), device='cuda')
    # warmup
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return min(times)


def time_decode(model, name, n_steps=DECODE_STEPS):
    model.eval()
    x = torch.randint(0, VOCAB, (1, 32), device='cuda')
    # prefill first, then measure decode steps
    with torch.no_grad():
        for _ in range(3):
            model(x)  # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        cur = x
        for _ in range(n_steps):
            with torch.no_grad():
                lg, _ = model(cur)
            nxt = lg[:, -1:].argmax(dim=-1)
            cur = torch.cat([cur, nxt], dim=1)
        torch.cuda.synchronize()
        total = time.time() - t0
    return total / n_steps  # 每token秒


def main():
    device = 'cuda'
    if not torch.cuda.is_available():
        print("NO CUDA"); return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    models = {n: make(n).to(device) for n in ['fusedfw_full', 'bdh', 'tf']}

    print("\n=== PREFILL（前向，batch=2，扫T）===")
    print(f"{'架构':<14}{'T=256':>10}{'512':>10}{'1024':>10}{'2048':>10}  {'翻倍比(1024→2048)'}")
    prefill = {}
    for name, m in models.items():
        row = []
        for T in T_LIST:
            t = time_prefill(m, name, T)
            row.append(t)
        prefill[name] = dict(zip(T_LIST, row))
        ratio = row[-1] / row[-2] if len(row) >= 2 else float('nan')
        print(f"{name:<14}" + "".join(f"{v*1000:>10.1f}" for v in row) + f"  {ratio:>10.2f}")

    print("\n=== DECODE（逐token，每token ms）===")
    print(f"{'架构':<14}{'每token ms':>12}")
    for name, m in models.items():
        ms = time_decode(m, name) * 1000
        print(f"{name:<14}{ms:>12.2f}")


if __name__ == '__main__':
    main()
