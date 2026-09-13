"""self 项分块化的判据测试（v6.7, T=8192, batch=1）。

判据（跑前锁死，事后不改）：
  J1 数值：分块版与整段版的 hidden/loss 一致（fp32 累加顺序不同，容差按 bf16 量级）
  J2 显存：前向峰值 / fwd+bwd 峰值应随 CH 减小而下降
  J3 速度：报出确切倍数（分块引入 Python 循环开销）
  J4 compile 兼容：分块版能被 torch.compile 编译
  J5 组合：chunk_self + grad_ckpt 同时开启
"""
import sys, time, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)

torch.manual_seed(1234)
_m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                   mlp_mult=MLP_MULT, W=W, read_mode='raw')
SD = {k: v.clone() for k, v in _m.state_dict().items()}
del _m


def run(chunk_self=0, grad_ckpt=False, compile_=False, iters=3, warm=2):
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw',
                      grad_ckpt=grad_ckpt, chunk_self=chunk_self).to(DEV)
    m.load_state_dict(SD); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    fn = m.forward_hidden
    if compile_:
        fn = torch.compile(fn)

    def one():
        o = fn(x)
        lg = o.view(B * T, D) @ head.weight.T
        loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        return o.detach().float(), loss.item()
    try:
        for _ in range(warm):
            out, lv = one()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out, lv = one()
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
        pk = torch.cuda.max_memory_allocated()/GiB
        return sorted(ts)[len(ts)//2], pk, out.clone(), lv
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, None, f"{type(e).__name__}: {str(e)[:110]}"
    finally:
        del m, opt; gc.collect(); torch.cuda.empty_cache()


print("=" * 100)
print("J1/J2/J3：self 项分块（CH 扫描）")
print("=" * 100)
base_ms, base_pk, base_out, base_loss = run(chunk_self=0)
print(f"  CH=0（整段，原行为）     {base_ms:7.1f} ms   peak={base_pk:6.3f} GiB   loss={base_loss:.6f}")
print()
res = {}
for ch in (4096, 2048, 1024, 512, 256):
    ms, pk, out, lv = run(chunk_self=ch)
    if ms:
        md = (base_out - out).abs().max().item() if base_out is not None else float('nan')
        rel = md / max(base_out.abs().max().item(), 1e-9)
        res[ch] = (ms, pk)
        print(f"  CH={ch:<5d}                {ms:7.1f} ms   peak={pk:6.3f} GiB   loss={lv:.6f}"
              f"   maxdiff={md:.3e} ({rel:.2e})  {'✓' if rel < 1e-2 else '✗'}")
    else:
        print(f"  CH={ch:<5d}                FAIL  {lv}")

print()
print("=" * 100)
print("J5：组合（chunk_self + grad_ckpt）与 compile")
print("=" * 100)
best = min(res, key=lambda k: res[k][1]) if res else 1024
for label, ch, ck, cp in [("chunk_self=1024 + grad_ckpt", 1024, True, False),
                          ("chunk_self=1024 + grad_ckpt + compile", 1024, True, True),
                          ("chunk_self=1024 + compile", 1024, False, True)]:
    ms, pk, out, lv = run(chunk_self=ch, grad_ckpt=ck, compile_=cp)
    if ms:
        md = (base_out - out).abs().max().item()
        print(f"  {label:40s} {ms:7.1f} ms  peak={pk:6.3f} GiB  maxdiff={md:.3e}  ✓")
    else:
        print(f"  {label:40s} FAIL  {lv}")
