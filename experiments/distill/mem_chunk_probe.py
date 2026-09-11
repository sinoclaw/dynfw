"""Chunked CE 显存剖析 v2（方案1 可行性实测）
口径：V=151936(Qwen3-0.6B), D=1024, B=8, T=1024, fp32 CE, reduction=batchmean, 固定输入
校验：所有分块实现的 loss 必须与全量数值一致（分块不得改变数学）
"""
import torch, torch.nn.functional as F
import torch.utils.checkpoint as C
import json, time

V, D = 151936, 1024
B, T = 8, 1024
TEMP = 1.0
dev = 'cuda'
torch.manual_seed(0)   # 固定 seed：所有方案共用同一份随机输入

h_s = torch.randn(B, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
h_t = torch.randn(B, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
W_s = torch.nn.Linear(D, V, bias=False, device=dev, dtype=torch.bfloat16)
W_t = torch.nn.Linear(D, V, bias=False, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    T_LG = W_t(h_t).float()          # 固定教师 logits（离线蒸馏场景）
print(f"教师 logits 张量本身: {T_LG.numel()*4/2**30:.2f} GiB", flush=True)


def kl_full(s_lg):
    ls = F.log_softmax(s_lg / TEMP, dim=-1)
    p = F.softmax(T_LG / TEMP, dim=-1)
    return F.kl_div(ls, p, reduction='batchmean') * (TEMP ** 2)


def kl_chunked(chunk, ckpt, teacher_online):
    """沿 T 分块。teacher_online=True 时教师 lm_head 也在块内现算（在线蒸馏，不物化全量教师 logits）。"""
    ntok = h_s.size(1)
    total = 0.0
    def seg(hs, ht):
        s = W_s(hs).float()
        t = W_t(ht).float() if teacher_online else ht
        ls = F.log_softmax(s / TEMP, dim=-1)
        p = F.softmax(t / TEMP, dim=-1)
        return F.kl_div(ls, p, reduction='sum') * (TEMP ** 2)
    for i in range(0, ntok, chunk):
        hs = h_s[:, i:i + chunk]
        if ckpt:
            part = C.checkpoint(seg, hs, T_LG[:, i:i + chunk] if not teacher_online else h_t[:, i:i + chunk],
                                use_reentrant=False)
        else:
            part = seg(hs, T_LG[:, i:i + chunk]) if not teacher_online else seg(hs, h_t[:, i:i + chunk])
        total = total + part
    return total / B


def measure(label, fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    loss = fn()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"{label:34s} peak={peak:7.2f} GiB  loss={loss.item():.6f}  {dt:5.1f}s", flush=True)
    return {'label': label, 'peak_gib': round(peak, 2), 'loss': round(loss.item(), 6), 'sec': round(dt, 1)}


res = []
res.append(measure("A 全量 B*T*V fp32 (现状)", lambda: kl_full(W_s(h_s).float())))
res.append(measure("B T分块256 无ckpt", lambda: kl_chunked(256, False, False)))
res.append(measure("C T分块256 +ckpt", lambda: kl_chunked(256, True, False)))
res.append(measure("D T分块64  +ckpt", lambda: kl_chunked(64, True, False)))
res.append(measure("E T分块64  +ckpt +教师在线", lambda: kl_chunked(64, True, True)))

base = res[0]['loss']
print()
print("数值等价校验（相对 A 的偏差）:")
for r in res:
    print(f"  {r['label']:34s} delta={abs(r['loss']-base)/max(abs(base),1e-9):.3e}")
print("SUMMARY_JSON " + json.dumps(res, ensure_ascii=False))
print(f"GPU total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB, "
      f"used_now={torch.cuda.memory_allocated()/2**30:.2f} GiB")
