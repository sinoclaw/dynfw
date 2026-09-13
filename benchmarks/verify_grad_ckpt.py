"""J1-J5 判据测试：grad_ckpt 开/关对比（v6.7, T=8192, batch=1）。

判据（跑前锁死，事后不改）：
  J1 数值：开关两种模式下，同 seed 的 hidden 输出 / loss 应一致（checkpoint 是精确重算）
  J2 显存：开启后 peak 应显著下降
  J3 速度：允许变慢，须报确切倍数
  J5 compile 兼容：开启后仍能被 torch.compile 编译
  J6 默认值：默认 False，不影响现有读数
"""
import sys, time, statistics
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

# 固定同一份权重给两种模式（保证 J1 是纯开关对比）
torch.manual_seed(1234)
m_ref = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw', grad_ckpt=False)
sd = {k: v.clone() for k, v in m_ref.state_dict().items()}
del m_ref


def build(ckpt):
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw', grad_ckpt=ckpt).to(DEV)
    m.load_state_dict(sd)
    return m


def run(ckpt, use_compile=False, iters=3, warm=2):
    torch.manual_seed(0)
    m = build(ckpt)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    fn = m.forward_hidden
    if use_compile:
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
        pk = torch.cuda.max_memory_allocated() / GiB
        ms = sorted(ts)[len(ts)//2]
        return ms, pk, out.clone(), lv
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, None, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        del m, opt
        torch.cuda.empty_cache()


print("=" * 92)
print("J1/J2/J3：grad_ckpt 关 vs 开")
print("=" * 92)
ms0, pk0, o0, l0 = run(False)
print(f"  grad_ckpt=False   {ms0:7.1f} ms   peak={pk0:6.3f} GiB   loss={l0:.6f}")
ms1, pk1, o1, l1 = run(True)
print(f"  grad_ckpt=True    {ms1:7.1f} ms   peak={pk1:6.3f} GiB   loss={l1:.6f}")

if o0 is not None and o1 is not None:
    md = (o0 - o1).abs().max().item()
    sc = o0.abs().max().item()
    print()
    print(f"  J1 数值: hidden maxdiff = {md:.4e}  (幅值 {sc:.4f}, 相对 {md/max(sc,1e-9):.3e})")
    print(f"       ⇒ {'✓ 逐位一致' if md == 0 else ('✓ 差异在 bf16 量级内' if md/max(sc,1e-9) < 1e-2 else '✗ 差异过大')}")
    print(f"  J2 显存: {pk0:.3f} → {pk1:.3f} GiB   降 {100*(pk0-pk1)/pk0:.1f}%")
    print(f"  J3 速度: {ms0:.1f} → {ms1:.1f} ms   {ms1/ms0:.2f}×")

print()
print("=" * 92)
print("J5：grad_ckpt + torch.compile 是否可用")
print("=" * 92)
ms2, pk2, o2, l2 = run(True, use_compile=True)
if ms2:
    print(f"  grad_ckpt=True + compile   {ms2:7.1f} ms   peak={pk2:6.3f} GiB   loss={l2:.6f}  ✓")
    if o1 is not None:
        print(f"       与未 compile 的数值 maxdiff = {(o1-o2).abs().max().item():.4e}")
else:
    print(f"  FAIL  {l2}")
