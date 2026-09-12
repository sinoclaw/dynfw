"""编译版残差归因 + 多采样稳态复测。

用法: PYTHONPATH=/data/dynfw python benchmarks/profile_comp.py
"""
import sys, time
import torch
from torch.profiler import profile, ProfilerActivity

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


def mk(arch):
    if arch == 'tf':
        return torch.compile(TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T))
    torch.manual_seed(0)
    m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)
    return torch.compile(to_opt5(m, True, False))


def main():
    B = 2
    for T in (8192, 32768):
        print()
        print('#' * 92)
        print(f'### T={T} batch={B}')
        print('#' * 92)
        for arch in ('tf', 'opt5'):
            m = mk(arch).cuda().train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.randint(0, VOCAB, (B, T), device='cuda')
            y = torch.randint(0, VOCAB, (B, T), device='cuda')

            def one():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, l = m(x, y)
                l.backward(); opt.step(); opt.zero_grad(set_to_none=True)

            # 多采样取稳态：1 次预热 + 3 轮各 3 次，报中位与最小
            for _ in range(2):
                one()
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                t0 = time.time()
                for _ in range(3):
                    one()
                torch.cuda.synchronize()
                samples.append(1000 * (time.time() - t0) / 3)
            samples.sort()
            print(f'\n--- {arch}  中位 {samples[1]:.1f}ms  最小 {samples[0]:.1f}ms  '
                  f'最大 {samples[2]:.1f}ms  样本 {[round(s,1) for s in samples]}')

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                one(); torch.cuda.synchronize()
            ka = [e for e in p.key_averages(group_by_input_shape=True)
                  if e.self_device_time_total > 0]
            ka.sort(key=lambda e: -e.self_device_time_total)
            tot = sum(e.self_device_time_total for e in ka)
            for e in ka[:14]:
                shp = str(e.input_shapes)[:46].replace(' ', '')
                print(f'{e.self_device_time_total/tot:>6.1%}{e.self_device_time_total/1000:>8.2f}ms'
                      f'{e.count:>6}  {e.key[:38]:<40}{shp}')
            del m, opt, x, y
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
