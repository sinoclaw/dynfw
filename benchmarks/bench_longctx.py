"""长上下文对照：TF(放开 maxT) vs v6w(W=256/W=1024)，只在 T≫W 区看真实拐点。

必须放开 TF 的 maxT —— 基线 transformer.py 里 MAXT=8192 的位置编码正弦表
会让 T>8192 直接报 tensor size mismatch，导致误判成「TF 更快」。
"""
import sys, time, math, argparse
import torch

sys.path.insert(0, '/data/dynfw')
VOCAB, D, NH, NL = 50257, 256, 8, 6
BIG_T = 262144


def build(arch, W=256):
    if arch == 'v6w':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W), 3 * 4 * D * D * NL
    from dynfw.models.transformer import TF_sdpa
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T), 12 * D * D * NL


def attn_macs(arch, T, W, B):
    """注意力部分 MACs(每步)，用于算 FLOPs 归一化的相对成本"""
    if arch == 'v6w':
        # 每 chunk: q@kT (w*w*N) + retr (w*N*D) + 累积 (w*N*D)，×nh×B×chunks(=T/w)×层
        return B * NH * (T / W) * (W * W * (4 * D // NH) + 2 * W * (4 * D // NH) * D) * NL
    return B * NH * T * T * (D // NH) * NL


def bench(m, T, B, iters=3, warmup=1):
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    try:
        for _ in range(warmup):
            one()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9, None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); return None, None, 'OOM'
    except Exception as e:
        torch.cuda.empty_cache(); return None, None, str(e)[:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--Ts', default='4096,8192,16384,32768')
    a = ap.parse_args()
    Ts = [int(t) for t in a.Ts.split(',')]

    for arch, W in [('tf', None), ('v6w', 256), ('v6w', 1024)]:
        label = arch if W is None else f'{arch}(W={W})'
        print(f"\n{'='*78}\n### {label}  batch={a.batch}  D={D} nh={NH} L={NL}\n{'='*78}")
        print(f"{'T':>8}{'ms/步':>11}{'显存GB':>10}{'注意力MACs':>15}{'ms/GF-MACs':>13}{'备注':>9}")
        for T in Ts:
            m, _ = build(arch, W)
            m = m.cuda()
            dt, mem, err = bench(m, T, a.batch)
            mac = attn_macs(arch, T, W, a.batch)
            if err:
                print(f"{T:>8}{'--':>11}{'--':>10}{mac/1e9:>14.1f}G{'--':>13}{err:>9}")
            else:
                print(f"{T:>8}{dt*1e3:>10.1f}{mem:>10.2f}{mac/1e9:>14.1f}G"
                      f"{dt*1e3/(mac/1e9):>13.4f}{'':>9}")
            del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
