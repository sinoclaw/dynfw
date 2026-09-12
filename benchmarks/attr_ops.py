"""aten op 级 CUDA 归因：TF vs 基线 vs OPT2，定位剩余 2.14x 差在哪。

用法: PYTHONPATH=/data/dynfw python benchmarks/attr_ops.py --T 8192 --batch 2
"""
import sys, time, argparse
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt2
from dynfw.models.transformer import TF_sdpa

D, NH, NL, W, V = 256, 8, 6, 256, 50257
BIG_T = 262144


def mk(arch):
    if arch == 'tf':
        return TF_sdpa(D=D, nh=NH, vocab=V, n_layer=NL, maxT=BIG_T).cuda().train()
    base = BDHBlockFWCycleLM(D=D, nh=NH, vocab=V, n_layer=NL, steps=1, mlp_mult=4, W=W)
    if arch == 'opt2':
        base = to_opt2(base, strict_bf16=True)
    return base.cuda().train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--archs', default='tf,base,opt2')
    a = ap.parse_args()
    B, T = a.batch, a.T

    for arch in a.archs.split(','):
        m = mk(arch)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, V, (B, T), device='cuda')
        y = torch.randint(0, V, (B, T), device='cuda')

        def one():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, l = m(x, y)
            l.backward(); opt.step(); opt.zero_grad(set_to_none=True)

        one(); one(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            one()
        torch.cuda.synchronize()
        ms = 1000 * (time.time() - t0) / 3

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            one(); torch.cuda.synchronize()
        rows = [(e.self_device_time_total, e.key, e.count) for e in p.key_averages()
                if e.self_device_time_total > 0]
        rows.sort(reverse=True)
        tot = sum(r[0] for r in rows)

        print('=' * 92)
        print(f'### {arch}  T={T} B={B}  {ms:.1f} ms/step   (GPU self total {tot/1000:.1f} ms)')
        print('=' * 92)
        for us, k, c in rows[:20]:
            print(f'{us/tot:>6.1%}{us/1000:>9.2f}ms{c:>7}  {k[:64]}')
        print()
        del m, opt, x, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
