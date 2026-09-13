"""验证 torch.utils.checkpoint 与 FLA kernel / compile 的兼容性（动手前先探）。
不通过就不改主文件。
"""
import sys, time, collections
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)


def run(use_ckpt, use_compile=False, iters=3):
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw').to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    if use_ckpt:
        import types
        def fwd_hidden_ckpt(self, x):
            blks = self.blocks
            for blk in blks:
                # 只对 block 的 (x, memories) -> (x, memories) 做 checkpoint
                orig = blk.forward
                def wrapped(xx, mems=None, _f=orig):
                    return _f(xx, mems)
                x = checkpoint(wrapped, x, None, use_reentrant=False)[0]
            return self.ln(x)
        m.forward_hidden = types.MethodType(fwd_hidden_ckpt, m)

    fn = m.forward_hidden
    if use_compile:
        fn = torch.compile(fn)

    def one():
        o = fn(x)
        lg = o.view(B * T, D) @ head.weight.T
        loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        return o.detach().float().mean().item(), loss.item()

    try:
        for _ in range(2):
            om, lv = one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            om, lv = one()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        pk = torch.cuda.max_memory_allocated() / GiB
        ms = sorted(ts)[len(ts) // 2]
        return ms, pk, om, lv
    except Exception as e:
        return None, None, f"{type(e).__name__}: {str(e)[:110]}", None
    finally:
        del m, opt
        torch.cuda.empty_cache()


print("=" * 92)
print("探针：checkpoint 与 FLA / compile 是否兼容")
print("=" * 92)
for label, ck, cp in [("baseline（无 ckpt, 无 compile）", False, False),
                      ("checkpoint（无 compile）", True, False),
                      ("baseline + compile", False, True),
                      ("checkpoint + compile", True, True)]:
    ms, pk, om, lv = run(ck, cp)
    if ms:
        print(f"  {label:34s} {ms:8.1f} ms   peak={pk:6.3f} GiB   out_mean={om:.5f}  loss={lv:.1f}")
    else:
        print(f"  {label:34s} FAIL  {om}")
