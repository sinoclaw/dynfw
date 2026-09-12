"""v6.7 vs TF 速度构成拆解 —— 69.0ms vs 45.4ms，差在哪。

方法：torch.profiler（CUDA 活动），按算子类别聚合 CPU/CUDA 时间；
     并逐段手测 attention 内部各步骤（RoPE / permute / FLA kernel / 块内 raw）。
"""
import sys, time, collections
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
torch.backends.cuda.matmul.allow_tf32 = True

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head_small = torch.nn.Linear(D, 4096, bias=False).to(DEV)


def build(arch):
    if arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
    raise ValueError(arch)


def one_step(m, opt):
    o = m.forward_hidden(x)
    lg = o.view(B * T, D) @ head_small.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)


# ================= 1) profiler 算子级聚合 =================
for arch in ('tf', 'v6.7'):
    m = build(arch)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(3):
        one_step(m, opt)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(3):
            one_step(m, opt)
        torch.cuda.synchronize()

    agg = collections.defaultdict(float)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            agg[e.key] += e.self_device_time_total / 3.0   # 3 steps
    total = sum(agg.values())
    print('=' * 78)
    print(f'{arch}  CUDA 算子总时间/step = {total/1000:.1f} ms（top 12）')
    print('=' * 78)
    for k, v in sorted(agg.items(), key=lambda t: -t[1])[:12]:
        print(f'  {v/1000:8.2f} ms  {v/total*100:5.1f}%  {k[:62]}')
    print(f'  ... 其余 {len(agg)-12} 个算子合计 {(total-sum(v for _,v in sorted(agg.items(),key=lambda t:-t[1])[:12]))/1000:.2f} ms')
    del m, opt
    torch.cuda.empty_cache()
    print()

# ================= 2) v6.7 attention 内部逐段手测 =================
from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                  W=W, read_mode='raw').to(DEV)
attn = m.blocks[0].attn
N = attn.N
Q = torch.randn(B, NH, T, N, device=DEV)
V = torch.randn(B, 1, T, D, device=DEV)


def tmed(fn, n=8, w=3):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2]


with torch.no_grad():
    r = torch.arange(0, T, device=DEV, dtype=torch.float32).view(1, 1, -1, 1)
    rope = lambda: attn.rope(r * attn.freqs, Q)
    perm = lambda: Q.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    full = lambda: attn(Q, Q, V, memories=None, W=W)

    print('=' * 78)
    print('v6.7 attention 内部逐段（T=8192, no_grad, 中位）')
    print('=' * 78)
    print(f'  RoPE (rope)                       {tmed(rope):8.2f} ms')
    print(f'  permute+contiguous+bf16 (1 个)    {tmed(perm):8.2f} ms')
    print(f'  整个 attn 前向 (fused, 含 4 次搬运) {tmed(full):8.2f} ms  ⇒ 搬运占约 {4*tmed(perm)/tmed(full)*100:.0f}%')
