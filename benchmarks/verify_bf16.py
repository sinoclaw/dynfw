"""bf16 混合精度验收（GPU 一到就跑）。

背景：学生模型此前全程 fp32（distill_qwen.py 里 bfloat16 只给教师模型）。
本次加 --bf16（torch.autocast）。⚠️ TF 同为 fp32 ⇒ 历史对比同口径、排名有效，
但所有绝对数字虚高；双方都改 bf16 后 TF 的 flash kernel 才会真正打开 ⇒ 谁获益多必须实测。

判据（跑前锁死，事后不改）：
  J1 数值：bf16 vs fp32 的 hidden/loss 差异在 bf16 容差内（fp32 版为基准）
  J2 显存：bf16 峰值应显著下降（目标接近减半）
  J3 速度：bf16 应更快（报确切倍数）
  J4 公平：TF 也测 bf16 ⇒ 报「相对差距在 fp32 与 bf16 下各是多少」，不预设我们获益更多
  J5 组合：bf16 + grad_ckpt 共存
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

torch.manual_seed(1234)
_ = torch.empty(0)


def bench(name, build, bf16_on, iters=3, warm=2):
    torch.manual_seed(0)
    m = build()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    head = torch.nn.Linear(D, 4096, bias=False).to(DEV)

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16_on):
            o = m.forward_hidden(x)
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
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        pk = torch.cuda.max_memory_allocated() / GiB
        return sorted(ts)[len(ts) // 2], pk, out.clone(), lv
    finally:
        del m, opt, head; gc.collect(); torch.cuda.empty_cache()


x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)


def parse_size():
    pass


# ---- 收集 v6.7 与 TF 的 fp32 基准 ----
from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
from dynfw.models.transformer import TF_sdpa

SD_FLA = None


def build_fla(ckpt=False):
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw', grad_ckpt=ckpt).to(DEV)
    return m


def build_tf():
    return TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)


print("=" * 100)
print("J1/J2/J3/J4：bf16 混合精度 —— v6.7 与 TF 双方同测")
print("=" * 100)
res = {}
for tag, bld in [('v6.7(FLA)', lambda: build_fla(False)), ('TF(sdpa)', build_tf)]:
    ms0, pk0, o0, l0 = bench(tag, bld, False)
    ms1, pk1, o1, l1 = bench(tag, bld, True)
    md = (o0 - o1).abs().max().item()
    rel = md / max(o0.abs().max().item(), 1e-9)
    res[tag] = dict(ms0=ms0, pk0=pk0, ms1=ms1, pk1=pk1, l0=l0, l1=l1, rel=rel)
    print(f"  {tag:12s} fp32: {ms0:7.1f} ms  {pk0:6.3f} GiB  loss={l0:.5f}")
    print(f"  {'':12s} bf16: {ms1:7.1f} ms  {pk1:6.3f} GiB  loss={l1:.5f}")
    print(f"  {'':12s} J1 数值 rel={rel:.2e} {'✓' if rel < 5e-2 else '✗'}  "
          f"J2 显存 -{100*(pk0-pk1)/pk0:.1f}%  J3 速度 {ms0/ms1:.2f}×")
    print()

a, t = res['v6.7(FLA)'], res['TF(sdpa)']
print("J4 公平对比（相对差距）")
print(f"  速度 v6.7/TF :  fp32 {a['ms0']/t['ms0']:.2f}×  →  bf16 {a['ms1']/t['ms1']:.2f}×")
print(f"  显存 v6.7/TF :  fp32 {a['pk0']/t['pk0']:.2f}×  →  bf16 {a['pk1']/t['pk1']:.2f}×")
print()

print("=" * 100)
print("J5：bf16 + grad_ckpt 组合")
print("=" * 100)
ms, pk, o, l = bench('v6.7 bf16+ckpt', lambda: build_fla(True), True)
print(f"  bf16 + grad_ckpt   {ms:7.1f} ms   peak={pk:6.3f} GiB   loss={l:.5f}  ✓")
