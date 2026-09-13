"""rope_fast 验收（GPU 一到就跑）。

判据（跑前锁死）：
  J1 数值：rope_fast vs rope 的 hidden/loss 逐位一致（数学等价，仅分配方式不同）
  J2 显存：峰值应下降（RoPE 在峰值时刻占约 0.5 GiB）
  J3 速度：报确切倍数
  J4 组合：rope_fast + grad_ckpt + bf16 三者共存
"""
import sys, time, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'; T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)

torch.manual_seed(1234)
from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
_m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT, W=W, read_mode='raw')
SD = {k: v.clone() for k, v in _m.state_dict().items()}
del _m


def run(rope_fast=False, grad_ckpt=False, bf16=False, iters=3, warm=2):
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                      W=W, read_mode='raw', grad_ckpt=grad_ckpt, rope_fast=rope_fast).to(DEV)
    m.load_state_dict(SD); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            o = m.forward_hidden(x)
            lg = o.view(B * T, D) @ head.weight.T
            loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        return o.detach().float(), loss.item()
    try:
        for _ in range(warm): out, lv = one()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out, lv = one()
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
        return sorted(ts)[len(ts)//2], torch.cuda.max_memory_allocated()/GiB, out.clone(), lv
    finally:
        del m, opt; gc.collect(); torch.cuda.empty_cache()


print("=" * 96); print("J1/J2/J3：rope 原版 vs rope_fast"); print("=" * 96)
ms0, pk0, o0, l0 = run(False)
ms1, pk1, o1, l1 = run(True)
md = (o0 - o1).abs().max().item()
print(f"  rope     : {ms0:7.1f} ms  peak={pk0:6.3f} GiB  loss={l0:.6f}")
print(f"  rope_fast: {ms1:7.1f} ms  peak={pk1:6.3f} GiB  loss={l1:.6f}")
print(f"  J1 maxdiff={md:.4e} {'✓ 逐位一致' if md == 0 else ('✓ bf16 量级' if md/max(o0.abs().max().item(),1e-9) < 1e-2 else '✗ 差异过大')}")
print(f"  J2 显存 {pk0:.3f} → {pk1:.3f} GiB  降 {100*(pk0-pk1)/pk0:.1f}%")
print(f"  J3 速度 {ms0/ms1:.2f}×")
print()
print("=" * 96); print("J4：组合（rope_fast + grad_ckpt + bf16）"); print("=" * 96)
ms, pk, o, l = run(True, True, True)
print(f"  三者全开  {ms:7.1f} ms  peak={pk:6.3f} GiB  loss={l:.6f}")
