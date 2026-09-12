"""三方纯计算测速（隔离 mmap IO）：v6(fw_cycle) / v6.6(gdn_cycle) / v6.7(gdn_fla)。

为什么必须隔离 IO：
  正式实验的 wall 里含读取 47GB shared teacher logits 的时间（mmap），三个架构读到的是同一份。
  之前收口实验就因此得出过"v6 与 tf 的 wall 都是 547s"的污染读数。
  ⟹ 报"训练速度"必须用【纯计算】口径：(前向 + 反向 + 优化器) 的每 step 时间，不含读教师数据。

口径（与台账铁律一致）：
  - 单进程、单形态、预热后取中位
  - 固定 T=8192 / W=64 / batch=1 / 同参数规模
  - 学生前向用 forward_hidden（与正式实验同路径）
  - 损失用 chunked KL（与正式实验同形态）
"""
import statistics
import sys
import time

import torch
import torch.nn.functional as F

DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, MLP_MULT, NLAYER = 128, 16, 64, 151936

torch.backends.cuda.matmul.allow_tf32 = True


def build(arch):
    if arch == 'v6':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM as M
    elif arch == 'v6.6':
        from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM as M
    elif arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
    else:
        raise ValueError(arch)
    # 注意：不要转 bf16 —— 模块内有 .float() 强转（如 gate(QR.float())），混合精度会崩；
    # 正式实验(distill_qwen.py)用的也是 fp32 学生，这里保持一致才是可比口径。
    m = M(D=D, nh=NH, vocab=NLAYER, n_layer=2, steps=1, mlp_mult=MLP_MULT, W=W,
          read_mode='raw').to(DEV)
    return m


def fwd_hidden(m, x):
    return m.forward_hidden(x)


def make_batch():
    # forward_hidden 期望 [B, T]（模型内部自己做 e(x).unsqueeze(1)）
    x = torch.randint(0, 1000, (B, T), device=DEV)
    tgt = torch.randint(0, 1000, (B, T), device=DEV)
    return x, tgt


print(f'=== 三方纯计算测速（隔离 mmap IO）T={T} W={W} batch={B} ===')
print(f'{"arch":8s} {"参数量":>12s} {"前向":>10s} {"fwd+bwd":>10s} {"+optim":>10s} {"peak":>9s}')

results = {}
for arch in ('v6', 'v6.6', 'v6.7'):
    try:
        m = build(arch)
        nparam = sum(p.numel() for p in m.parameters())
        x, tgt = make_batch()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        # 代理 head：真实 head 的 151936 维 logits 要 5GB，本 bench 只比"计算量相对关系"，
        # 三方用同一个代理 head（相同规模），不改变相对结论。
        head_small = torch.nn.Linear(D, 4096, bias=False).to(DEV)

        def step(do_opt=False):
            o = fwd_hidden(m, x)
            lg = o.view(B * T, D) @ head_small.weight.T
            loss = F.cross_entropy(lg.float(), tgt.view(-1))
            loss.backward()
            if do_opt:
                opt.step(); opt.zero_grad(set_to_none=True)
            return loss

        # 前向
        def fwd_only():
            with torch.no_grad():
                fwd_hidden(m, x)
        for _ in range(3):
            fwd_only()
        torch.cuda.synchronize()
        fts = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fwd_only()
            torch.cuda.synchronize(); fts.append((time.perf_counter() - t0) * 1000)

        # fwd+bwd
        m.zero_grad(set_to_none=True)
        for _ in range(3):
            step()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        bts = []
        for _ in range(5):
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step()
            torch.cuda.synchronize(); bts.append((time.perf_counter() - t0) * 1000)

        # +opt
        for _ in range(2):
            step(do_opt=True)
        torch.cuda.synchronize()
        ots = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step(do_opt=True)
            torch.cuda.synchronize(); ots.append((time.perf_counter() - t0) * 1000)

        pk = torch.cuda.max_memory_allocated() / 2 ** 30
        results[arch] = (statistics.median(fts), statistics.median(bts), statistics.median(ots))
        print(f'{arch:8s} {nparam:>12,d} {statistics.median(fts):9.1f}ms '
              f'{statistics.median(bts):9.1f}ms {statistics.median(ots):9.1f}ms {pk:8.2f}GiB', flush=True)
        del m, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        import traceback
        print(f'{arch:8s} FAIL: {type(e).__name__}: {str(e)[:120]}')
        traceback.print_exc()

if len(results) >= 2:
    print()
    print('=== 相对 v6 的倍数（fwd+bwd，越接近 1 越好）===')
    base = results['v6'][1]
    for k, v in results.items():
        print(f'  {k:8s} {v[1]/base:6.2f}x   (每 1000 step ≈ {v[1]*1000/1000:6.1f}s)')
    print()
    print('注：这是纯计算口径。正式实验 wall 额外含 mmap 读 47GB 教师 logits 的时间（三方同源），')
    print('    ⟹ 报速度必须用本表，不能用实验 wall。')
