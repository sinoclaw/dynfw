"""v6.7 vs TF 纯计算对轰（隔离 mmap IO）—— 与台账 §4.2e/§4.2f 同口径。

为什么要单独测：
  正式实验的 wall 里含 mmap 读 47GB 教师 logits 的时间（v6.7 wall 536s 中约 537s 是 IO），
  ⟹ 报"训练速度"必须用【纯计算】口径：(前向 + 反向 + 优化器) 每 step 时间。

口径（台账铁律）：
  - 单进程、单形态、预热后取中位
  - T=8192 / W=64 / batch=1（同正式实验）
  - 学生前向走 forward_hidden（与正式实验同路径）
  - 损失用同一代理 head（真实 151936 维 logits 需 5GB，三方同规模代理不改相对结论）
  - 参数量：v6.7 与 TF 不等（TF 少 14.6%），本脚本逐项打印，不做等价假设
"""
import statistics, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')

DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
torch.backends.cuda.matmul.allow_tf32 = True


def build(arch):
    if arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'v6':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
    raise ValueError(arch)


print(f'=== v6.7 vs TF 纯计算对轰（隔离 mmap IO）T={T} W={W} batch={B} ===')
print(f'{"arch":8s} {"参数量":>12s} {"前向":>11s} {"fwd+bwd":>11s} {"+optim":>11s} {"peak":>9s}')

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head_small = torch.nn.Linear(D, 4096, bias=False).to(DEV)

results = {}
for arch in ('v6.7', 'v6', 'tf'):
    try:
        m = build(arch)
        nparam = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step(do_opt=False):
            o = m.forward_hidden(x)
            lg = o.view(B * T, D) @ head_small.weight.T
            loss = F.cross_entropy(lg.float(), tgt.view(-1))
            loss.backward()
            if do_opt:
                opt.step(); opt.zero_grad(set_to_none=True)

        def fwd_only():
            with torch.no_grad():
                m.forward_hidden(x)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            fwd_only()
        torch.cuda.synchronize()
        fts = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fwd_only()
            torch.cuda.synchronize(); fts.append((time.perf_counter() - t0) * 1000)

        m.zero_grad(set_to_none=True)
        for _ in range(3):
            step(); m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        bts = []
        for _ in range(5):
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step()
            torch.cuda.synchronize(); bts.append((time.perf_counter() - t0) * 1000)

        for _ in range(2):
            step(do_opt=True)
        torch.cuda.synchronize()
        ots = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step(do_opt=True)
            torch.cuda.synchronize(); ots.append((time.perf_counter() - t0) * 1000)

        pk = torch.cuda.max_memory_allocated() / 2 ** 30
        results[arch] = (statistics.median(fts), statistics.median(bts), statistics.median(ots), pk, nparam)
        print(f'{arch:8s} {nparam:>12,d} {statistics.median(fts):10.1f}ms '
              f'{statistics.median(bts):10.1f}ms {statistics.median(ots):10.1f}ms {pk:8.2f}GiB', flush=True)
        del m, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        import traceback
        print(f'{arch:8s} FAIL: {type(e).__name__}: {str(e)[:130]}')
        traceback.print_exc()

if 'v6.7' in results and 'tf' in results:
    a, b = results['v6.7'], results['tf']
    print()
    print('=== v6.7 vs TF（fwd+bwd 纯计算）===')
    print(f'  前向    v6.7 {a[0]:.1f}ms  vs  TF {b[0]:.1f}ms   ⇒ v6.7 {b[0]/a[0]:.2f}x')
    print(f'  fwd+bwd v6.7 {a[1]:.1f}ms  vs  TF {b[1]:.1f}ms   ⇒ v6.7 {b[1]/a[1]:.2f}x')
    print(f'  +optim  v6.7 {a[2]:.1f}ms  vs  TF {b[2]:.1f}ms   ⇒ v6.7 {b[2]/a[2]:.2f}x')
    print(f'  显存    v6.7 {a[3]:.2f}GiB vs TF {b[3]:.2f}GiB  ⇒ TF 省 {(1-a[3]/b[3])*100:.0f}%')
    print(f'  参数    v6.7 {a[4]:,}  vs  TF {b[4]:,}  ⇒ v6.7 多 {100*(a[4]-b[4])/b[4]:.1f}%')
    print()
    print('注：v6.7 参数量多 14.6%，本表不做等价假设，读数需并列参数量一起看。')
