"""OPT 版 op 级 profile（T=8192）——定位 SDPA 之后剩余开销在哪。
用法: PYTHONPATH=/data/dynfw python benchmarks/profile_opt.py
"""
import sys, time
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt, to_opt2
from dynfw.models.transformer import TF_sdpa

D, NH, NL, W, V = 256, 8, 6, 256, 50257
BIG_T = 262144


def run(arch, T, B=2, iters=3):
    if arch == 'tf':
        m = TF_sdpa(D=D, nh=NH, vocab=V, n_layer=NL, maxT=BIG_T).cuda().train()
    else:
        mk = lambda: BDHBlockFWCycleLM(D=D, nh=NH, vocab=V, n_layer=NL, steps=1, mlp_mult=4, W=W)
        if arch == 'opt':
            m = to_opt(mk(), strict_bf16=True).cuda().train()
        elif arch == 'opt2':
            m = to_opt2(mk(), strict_bf16=True).cuda().train()
        else:
            m = mk().cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, V, (B, T), device='cuda')
    y = torch.randint(0, V, (B, T), device='cuda')

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, l = m(x, y)
        l.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    one(); one(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / iters

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        one(); torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.key, e.count) for e in p.key_averages()
            if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)

    print('=' * 84)
    print(f'### {arch}  T={T} batch={B}  {ms:.1f} ms/step   (GPU 自耗时合计 {tot/1000:.1f}ms)')
    print('=' * 84)
    print(f"{'pct':>6}{'ms':>9}{'calls':>8}  kernel")
    for us, k, c in rows[:12]:
        print(f"{us/tot:>6.1%}{us/1000:>9.2f}{c:>8}  {k[:58]}")
    del m, opt, x, y
    torch.cuda.empty_cache()
    return ms


if __name__ == '__main__':
    import sys as _s
    _s.argv = _s.argv
    for arch in ('tf', 'base', 'opt', 'opt2'):
        run(arch, 32768)
