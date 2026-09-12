"""定位 OPT2 剩余开销的三个疑点：copy_ / bmm / fp32 sgemm 到底出自哪个 op 哪个 dtype。

用法: PYTHONPATH=/data/dynfw python benchmarks/probe_opt2.py
"""
import sys
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt2, FWAttentionOpt

V, D, NH, NL, W = 50257, 256, 8, 6, 256
B, T = 2, 8192


def part_a_shapes():
    """按 (op, 输入 shape) 分组，看 copy_ / bmm / sgemm 的实际形状。"""
    m = to_opt2(BDHBlockFWCycleLM(D=D, nh=NH, vocab=V, n_layer=NL, steps=1, mlp_mult=4, W=W),
                True).cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, V, (B, T), device='cuda')
    y = torch.randint(0, V, (B, T), device='cuda')

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, l = m(x, y)
        l.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    one(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as p:
        one(); torch.cuda.synchronize()

    ka = [e for e in p.key_averages(group_by_input_shape=True)
          if e.self_device_time_total > 0]
    ka.sort(key=lambda e: -e.self_device_time_total)
    tot = sum(e.self_device_time_total for e in ka)
    print('=' * 96)
    print(f'PART A — opt2 T={T} B={B} 按 (op,shape) 归因   GPU self 合计 {tot/1000:.1f}ms')
    print('=' * 96)
    for e in ka[:26]:
        shp = str(e.input_shapes)[:60].replace(' ', '')
        print(f'{e.self_device_time_total/tot:>6.1%}{e.self_device_time_total/1000:>8.2f}ms'
              f'{e.count:>6}  {e.key[:40]:<42}{shp}')
    del m, opt, x, y
    torch.cuda.empty_cache()


def part_b_dtypes():
    """逐 op 打出 dtype，找出谁在 autocast 下变 fp32。"""
    m = to_opt2(BDHBlockFWCycleLM(D=D, nh=NH, vocab=V, n_layer=NL, steps=1, mlp_mult=4, W=W),
                True).cuda().train()
    attn = m.blocks[0].attn
    N = D * 4 // NH          # mlp_mult=4 → N = 128
    print()
    print('=' * 96)
    print(f'PART B — dtype 追踪（autocast bf16）  N={N} nch={T//W}')
    print('=' * 96)

    def P(name, t):
        print(f'  {name:<26} {str(tuple(t.shape)):<26} {t.dtype}')

    Q = torch.randn(B, NH, T, N, device='cuda', dtype=torch.bfloat16)
    Vv = torch.randn(B, 1, T, D, device='cuda', dtype=torch.bfloat16)
    print(f'  {"attn.freqs":<26} {str(tuple(attn.freqs.shape)):<26} {attn.freqs.dtype}')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        r = torch.arange(0, T, device=attn.freqs.device, dtype=attn.freqs.dtype).view(1, 1, -1, 1)
        P('r * freqs', r * attn.freqs)
        QR = FWAttentionOpt.rope_fast(r * attn.freqs, Q)
        P('QR', QR)
        nch = T // W
        q = QR.view(B, NH, nch, W, N)
        v = Vv.view(B, 1, nch, W, D).expand(B, NH, nch, W, D)
        P('q', q); P('v (expand)', v)
        qf = q.reshape(B * NH * nch, W, N)
        vf = v.reshape(B * NH * nch, W, D)
        P('qf', qf); P('vf (expand+reshape)', vf)
        a = F.scaled_dot_product_attention(qf, qf, vf, is_causal=True, scale=1.0)
        P('sdpa a', a)
        kv = torch.einsum('bhcwd,bhcwe->bchde', q, v)
        P('einsum kv', kv)
        S = kv.cumsum(dim=1)
        P('cumsum', S)
        S = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
        P('cat excl', S)
        o = torch.einsum('bhcwd,bchde->bhcwe', q, S)
        P('einsum q@S', o)
        out = (a.view(B, NH, nch, W, D) + o).reshape(B, NH, T, D)
        P('out', out)


if __name__ == '__main__':
    part_a_shapes()
    part_b_dtypes()
