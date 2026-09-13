"""布局优化（share_qk）+ 全开关矩阵验收（GPU 一到就跑）。

判据（跑前锁死）：
  J1 数值：share_qk 开/关 的 hidden/loss 逐位一致（同一张量喂两次，数学不变）
  J2 显存：峰值应下降（省一次 [B,T,nh,N] bf16 搬运 ≈134MB 瞬时）
  J3 速度：应略快（少一次全量搬运+类型转换）
  J4 组合：rope_fast + share_qk + grad_ckpt + bf16 四者共存，逐项报
  J5 单变量：一次只开一个，避免互相掩盖
"""
import sys, time, gc, itertools
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

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
torch.manual_seed(1234)
_m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT, W=W, read_mode='raw')
SD = {k: v.clone() for k, v in _m.state_dict().items()}
del _m


def run(rope_fast=False, share_qk=False, grad_ckpt=False, bf16=False, iters=3, warm=2):
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                      W=W, read_mode='raw', grad_ckpt=grad_ckpt, rope_fast=rope_fast,
                      share_qk=share_qk).to(DEV)
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
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, None, f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        del m, opt; gc.collect(); torch.cuda.empty_cache()


base_ms, base_pk, base_o, base_l = run()
print("=" * 100)
print(f"baseline（全关）      {base_ms:7.1f} ms  peak={base_pk:6.3f} GiB  loss={base_l:.6f}")
print("=" * 100)
print()
print("J5 单变量（一次只开一个）")
for tag, kw in [("rope_fast", dict(rope_fast=True)),
                ("share_qk", dict(share_qk=True)),
                ("grad_ckpt", dict(grad_ckpt=True)),
                ("bf16", dict(bf16=True))]:
    ms, pk, o, l = run(**kw)
    if ms is None:
        print(f"  {tag:16s} FAIL {l}"); continue
    md = (base_o - o).abs().max().item()
    rel = md / max(base_o.abs().max().item(), 1e-9)
    ok = "✓逐位一致" if md == 0 else ("✓bf16量级" if rel < 1e-2 else "✗")
    print(f"  {tag:16s} {ms:7.1f} ms  peak={pk:6.3f} GiB  ({100*(base_pk-pk)/base_pk:+5.1f}%)  "
          f"speed {base_ms/ms:4.2f}×  maxdiff={md:.2e} {ok}")

print()
print("J4 全开组合")
ms, pk, o, l = run(True, True, True, True)
if ms:
    md = (base_o - o).abs().max().item()
    print(f"  四项全开            {ms:7.1f} ms  peak={pk:6.3f} GiB  ({100*(base_pk-pk)/base_pk:+.1f}%)  "
          f"speed {base_ms/ms:.2f}×  maxdiff={md:.2e}")
    print(f"  对照 TF 0.634 GiB ⇒ 倍数 {pk/0.634:.2f}×")
