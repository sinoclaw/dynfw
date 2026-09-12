"""op 级 profile：定位 v6 比 TF 慢 2.2× 的具体来源（实现问题，非算法复杂度）。
用法: PYTHONPATH=/data/dynfw python benchmarks/profile_fw_attn.py
"""
import sys, argparse
import torch

sys.path.insert(0, '/data/dynfw')
VOCAB, D, NH, NL = 50257, 256, 8, 6


def build(arch, W=256):
    if arch == 'v6w':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)
    from dynfw.models.transformer import TF_sdpa
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--topk', type=int, default=14)
    a = ap.parse_args()

    for arch, W in [('tf', None), ('v6w', 256)]:
        label = arch if W is None else f'{arch}(W={W})'
        m = (build(arch, W) if W else build(arch)).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, VOCAB, (a.batch, a.T), device='cuda')
        y = torch.randint(0, VOCAB, (a.batch, a.T), device='cuda')

        def step():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = m(x, y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 5

        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()

        print(f"\n{'='*80}\n### {label}   T={a.T} batch={a.batch}   {dt*1e3:.1f} ms/step\n{'='*80}")
        evs = prof.key_averages()
        rows = []
        for e in evs:
            if e.self_device_time_total > 0:
                rows.append((e.self_device_time_total, e.key, e.count))
        rows.sort(reverse=True)
        tot = sum(r[0] for r in rows)
        print(f"{'CUDA 自耗时占比':>16}  {'耗时ms':>9} {'调用次数':>9}  kernel")
        for us, k, c in rows[:a.topk]:
            print(f"{us/tot:>15.1%}  {us/1000/3:>9.2f} {c:>9}  {k[:60]}")
        print(f"{'合计':>16}  {tot/1000/3:>9.2f}")
        del m, opt, x, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
