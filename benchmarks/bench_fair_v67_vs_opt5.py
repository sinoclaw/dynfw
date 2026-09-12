"""① 公平性补齐：把 v6+opt5（正式交付形态）放进纯计算测速表，与 v6.7 对轰。

背景：
  上一版测速里 v6 是【未加 opt5】的基线（951ms/step），而正式实验用的 v6 是 opt5 形态（wall 538s）。
  拿"未优化的 v6"当基准会高估 v6.7 的优势 ⟹ 必须把 opt5 放进来对轰。

口径处理：
  to_opt5_raw(strict_bf16=True) 走 bf16 手写 bmm —— 但本测速是 fp32（v6.7 内部有 .float() 强转，
  bf16 会崩）。因此这里同时测两档：
    fp32 档: v6 / v6.6 / v6.7 / v6+opt5(strict_bf16=False)   ← 同口径可比
    bf16 档: v6(基线) / v6+opt5(strict_bf16=True)            ← 参考（v6.6/v6.7 无法跑 bf16）
  不混两档做结论。
"""
import statistics
import time

import torch
import torch.nn.functional as F

DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, MLP_MULT, VOCAB = 128, 16, 64, 151936


def build(arch, dtype):
    if arch in ('v6', 'v6+opt5'):
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM as M
    elif arch == 'v6.6':
        from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM as M
    elif arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
    else:
        raise ValueError(arch)
    m = M(D=D, nh=NH, vocab=VOCAB, n_layer=2, steps=1, mlp_mult=MLP_MULT, W=W,
          read_mode='raw').to(DEV)
    if dtype == torch.bfloat16:
        m = m.to(torch.bfloat16)
    if arch == 'v6+opt5':
        from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw
        to_opt5_raw(m, strict_bf16=(dtype == torch.bfloat16), bf16_prefix=False)
    return m


def run_suite(arch, dtype):
    m = build(arch, dtype)
    nparam = sum(p.numel() for p in m.parameters())
    head = torch.nn.Linear(D, 4096, bias=False).to(DEV).to(dtype)
    x = torch.randint(0, 1000, (B, T), device=DEV)
    tgt = torch.randint(0, 4096, (B, T), device=DEV)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-4)

    def step(do_opt=False):
        o = m.forward_hidden(x)
        lg = o.view(B * T, D) @ head.weight.T
        loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward()
        if do_opt:
            opt.step(); opt.zero_grad(set_to_none=True)

    def fwd_only():
        with torch.no_grad():
            m.forward_hidden(x)

    for _ in range(3): fwd_only()
    torch.cuda.synchronize()
    ft = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fwd_only()
        torch.cuda.synchronize(); ft.append((time.perf_counter() - t0) * 1000)

    m.zero_grad(set_to_none=True)
    for _ in range(3):
        step(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    bt = []
    for _ in range(5):
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter(); step()
        torch.cuda.synchronize(); bt.append((time.perf_counter() - t0) * 1000)

    for _ in range(2): step(do_opt=True)
    torch.cuda.synchronize()
    ot = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter(); step(do_opt=True)
        torch.cuda.synchronize(); ot.append((time.perf_counter() - t0) * 1000)

    pk = torch.cuda.max_memory_allocated() / 2 ** 30
    del m, opt, head
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return nparam, statistics.median(ft), statistics.median(bt), statistics.median(ot), pk


def suite(title, arch, dtype):
    print(f'--- {title} ---')
    print(f'{"arch":16s} {"参数量":>12s} {"前向":>9s} {"fwd+bwd":>10s} {"+optim":>9s} {"peak":>8s}')
    res = {}
    for a in arch:
        try:
            n, f, b, o, pk = run_suite(a, dtype)
            res[a] = b
            print(f'{a:16s} {n:>12,d} {f:8.1f}ms {b:9.1f}ms {o:8.1f}ms {pk:7.2f}GiB', flush=True)
        except Exception as e:
            print(f'{a:16s} FAIL: {type(e).__name__}: {str(e)[:90]}', flush=True)
    print()
    return res


print('=== ① 纯计算对轰（T=8192 W=64 batch=1，隔离 mmap IO）===')
print()
r32 = suite('fp32 档（同口径可比）', ['v6', 'v6.6', 'v6.7', 'v6+opt5'], torch.float32)
r16 = suite('bf16 档（参考；v6.6/v6.7 内部 fp32 强转，无法跑此档）', ['v6', 'v6+opt5'], torch.bfloat16)

print('=== 相对倍数（fwd+bwd）===')
for name, r in (('fp32', r32), ('bf16', r16)):
    if not r:
        continue
    base = r.get('v6')
    print(f'  [{name} 档] 以 v6({base:.1f}ms) 为 1.00x:')
    for k, v in r.items():
        print(f'    {k:16s} {v/base:5.2f}x   ({v:.1f}ms/step → 1000step≈{v:.0f}s)')
print()
print('注：1000 step 的纯计算时间下界；正式实验 wall 还含 mmap 读 47GB 教师 logits 的 IO（~537s）。')
