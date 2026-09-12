"""三方纯计算 + 显存对轰：TF / v6.6(gdn) / v6.7(FLA)。

口径与台账一致：T=8192 / W=64 / batch=1 / 同代理 head / 预热后取中位。
参数量逐项打印，不做等价假设（TF 少 14.6%）。
"""
import statistics, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True


def build(arch):
    if arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'v6.6':
        from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
    raise ValueError(arch)


print(f'=== 三方纯计算 + 显存对轰  T={T} W={W} batch={B} '
      f'D={D} nh={NH} n_layer={NLAYER} mlp_mult={MLP_MULT} ===')
print(f'{"arch":7s} {"参数量":>12s} {"前向":>10s} {"fwd+bwd":>10s} {"+optim":>10s} {"peak":>9s}')

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head_small = torch.nn.Linear(D, 4096, bias=False).to(DEV)

results = {}
for arch in ('v6.7', 'v6.6', 'tf'):
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

        pk = torch.cuda.max_memory_allocated() / GiB
        results[arch] = (statistics.median(fts), statistics.median(bts), statistics.median(ots), pk, nparam)
        print(f'{arch:7s} {nparam:>12,d} {statistics.median(fts):9.1f}ms '
              f'{statistics.median(bts):9.1f}ms {statistics.median(ots):9.1f}ms {pk:8.2f}GiB', flush=True)
        del m, opt
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        import traceback
        print(f'{arch:7s} FAIL: {type(e).__name__}: {str(e)[:130]}')
        traceback.print_exc()

if len(results) == 3:
    t, s, f = results['tf'], results['v6.6'], results['v6.7']
    print()
    print('=== 相对 TF 的倍数（>1 表示比 TF 慢 / 比 TF 费显存）===')
    print(f'{"":8s} {"前向":>12s} {"fwd+bwd":>12s} {"peak显存":>12s}')
    for nm, r in (('v6.6', s), ('v6.7', f)):
        print(f'{nm:8s} {r[0]/t[0]:11.2f}x {r[1]/t[1]:11.2f}x {r[3]/t[3]:11.2f}x')
    print()
    print('=== 相对 v6.6 的倍数（v6.7 的提速/省显存效果）===')
    print(f'  v6.7 vs v6.6:  前向 {f[0]/s[0]:.2f}x  fwd+bwd {f[1]/s[1]:.2f}x  显存 {f[3]/s[3]:.2f}x')
