"""FWAttention 实现优化对照：定位「慢 2.2×」里有多少是纯实现开销。

profile 已证：v6w 的 aten::copy_ 2610 次 / 38.2ms（TF 420次/10.4ms），
fill_ 783 次（每轮重建 mask），无融合注意力。

A 基线（现状）
B = A + 缓存 causal mask（数值应完全相同，maxdiff 必须 = 0）
C = B + torch.compile 整个 block（数值应几乎相同）
D = C + 去掉 einsum 的 .float() 强制转换（bf16 计算，记录数值差）

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_fw_opt.py
"""
import sys, time, copy
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import FWAttention, BDHBlockFWCycleLM

VOCAB, D, NH, NL = 50257, 256, 8, 6
_MASK = {}


# ---------- B: 缓存 causal mask ----------
def fwd_B(self, Q, K, V, memories=None, W=512):
    assert K is Q
    B, nh, T, N = Q.size()
    Dd = self.D
    r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
    QR = self.rope(r * self.freqs, Q)
    out_chunks = []
    new_mem = memories if memories is not None else torch.zeros(B, nh, N, Dd, device=Q.device)
    for st in range(0, T, W):
        en = min(st + W, T); w = en - st
        q_c = QR[:, :, st:en]; k_c = QR[:, :, st:en]; v_c = V[:, :, st:en]
        sim = q_c @ k_c.mT
        key = (w, Q.device)
        causal = _MASK.get(key)
        if causal is None:
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            _MASK[key] = causal
        sim = sim.masked_fill(~causal, float('-inf'))
        agg = torch.softmax(sim.float(), dim=-1) @ v_c
        if new_mem is not None and (st > 0 or memories is not None):
            agg = agg + torch.einsum('bhwd,bhde->bhwe', q_c.float(), new_mem.float())
        out_chunks.append(agg)
        new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c.float(), v_c.float())
    return torch.cat(out_chunks, dim=2), new_mem


# ---------- D: B + 去 fp32 强制转换 ----------
def fwd_D(self, Q, K, V, memories=None, W=512):
    assert K is Q
    B, nh, T, N = Q.size()
    Dd = self.D
    r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
    QR = self.rope(r * self.freqs, Q)
    out_chunks = []
    new_mem = memories if memories is not None else torch.zeros(B, nh, N, Dd, device=Q.device, dtype=Q.dtype)
    for st in range(0, T, W):
        en = min(st + W, T); w = en - st
        q_c = QR[:, :, st:en]; k_c = QR[:, :, st:en]; v_c = V[:, :, st:en]
        sim = q_c @ k_c.mT
        key = (w, Q.device)
        causal = _MASK.get(key)
        if causal is None:
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            _MASK[key] = causal
        sim = sim.masked_fill(~causal, float('-inf'))
        agg = torch.softmax(sim, dim=-1) @ v_c
        if new_mem is not None and (st > 0 or memories is not None):
            agg = agg + torch.einsum('bhwd,bhde->bhwe', q_c, new_mem)
        out_chunks.append(agg)
        new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c, v_c)
    return torch.cat(out_chunks, dim=2), new_mem


VARIANTS = [('A 基线', None), ('B +缓存mask', fwd_B), ('D +去fp32强转', fwd_D)]


def bench(m, T, B, iters=5, warmup=2):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')

    def one():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    try:
        for _ in range(warmup):
            one()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters, None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); return None, 'OOM'
    except Exception as e:
        torch.cuda.empty_cache(); return None, str(e)[:70]


def main():
    T, B = 1024, 8
    orig = FWAttention.forward
    ref_logits = {}
    print(f"T={T} batch={B}  D={D} nh={NH} L={NL}\n" + "=" * 72)
    print(f"{'变体':<18}{'ms/step':>10}{'相对A':>9}{'logits maxdiff vs A':>24}")
    base = None
    for name, fn in VARIANTS:
        FWAttention.forward = fn if fn else orig
        m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=256).cuda()
        # 数值对照（同一 init，用固定 seed 重建）
        torch.manual_seed(0)
        m2 = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=256).cuda()
        with torch.no_grad():
            xv = torch.randint(0, VOCAB, (1, 256), device='cuda')
            lg, _ = m2(xv, None)
        if name.startswith('A'):
            ref_logits['A'] = lg.float().clone()
            diff = 0.0
        else:
            diff = (lg.float() - ref_logits['A']).abs().max().item()
        del m2
        dt, err = bench(m, T, B)
        if err:
            print(f"{name:<18}{'--':>10}{'--':>9}{err:>24}")
        else:
            if base is None:
                base = dt
            print(f"{name:<18}{dt*1e3:>9.1f}{dt/base:>8.2f}x{diff:>24.3e}")
        del m; torch.cuda.empty_cache()
    FWAttention.forward = orig


if __name__ == '__main__':
    main()
