"""分块 KL 补充验证：
1) fp64 严格等价性（排除公式错误，只留精度差异）
2) 教师 logits 放 CPU 的显存收益（离线蒸馏场景天然可 CPU 常驻）
3) 反向相位峰值（是否真的比全量低）
"""
import sys, json, time, torch
sys.path.insert(0, '/data/dynfw')
from dynfw.models.transformer import TF_sdpa
from dynfw.training.chunked_kl import chunked_kl_loss, full_kl_loss

dev = 'cuda'
res = {}

# ---------- 1) fp64 严格等价 ----------
D, V, B, T = 64, 4096, 2, 64
torch.manual_seed(0)
h = torch.randn(B, T, D, device=dev, dtype=torch.float64, requires_grad=True)
W = (torch.randn(V, D, device=dev, dtype=torch.float64) * 0.02).requires_grad_(True)
b = torch.zeros(V, device=dev, dtype=torch.float64, requires_grad=True)
t = torch.randn(B, T, V, device=dev, dtype=torch.float64)
TEMP = 0.7

z = h @ W.t() + b
lA = full_kl_loss(z, t, TEMP)
lA.backward()
gA = {'h': h.grad.clone(), 'W': W.grad.clone(), 'b': b.grad.clone()}
for p in (h, W, b):
    p.grad = None

lB = chunked_kl_loss(h, W, b, t, chunk=8, temperature=TEMP)
lB.backward()
gB = {'h': h.grad.clone(), 'W': W.grad.clone(), 'b': b.grad.clone()}

d = {k: (gA[k] - gB[k]).abs().max().item() for k in gA}
print(f"[fp64] loss 全量={lA.item():.16f} 分块={lB.item():.16f} |Δ|={abs(lA.item()-lB.item()):.3e}")
for k in d:
    print(f"[fp64] grad[{k}] max|Δ|={d[k]:.3e}  max|g|={gA[k].abs().max().item():.3e}")
res['fp64'] = {'dloss': abs(lA.item() - lB.item()), 'dgrad': d}
del h, W, b, t, z, lA, lB
torch.cuda.empty_cache()

# ---------- 2) 教师 logits CPU vs GPU ----------
D, V, B, T, NH = 128, 151936, 8, 1024, 4
torch.manual_seed(0)
model = TF_sdpa(D=D, nh=NH, n_layer=2, vocab=V, maxT=T).to(dev)
x = torch.randint(0, V, (B, T), device=dev)
torch.manual_seed(2)
Wt = torch.nn.Linear(D, V, bias=False, device=dev, dtype=torch.float32)
with torch.no_grad():
    T_LG_GPU = Wt(model.forward_hidden(x).detach().float())
T_LG_CPU = T_LG_GPU.to('cpu')
W0, b0 = model.head_params()
print(f"\n教师 logits = {T_LG_GPU.numel()*4/2**30:.2f} GiB")


def run(t_lg, tag):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    h = model.forward_hidden(x)
    torch.cuda.synchronize(); peak_fw = torch.cuda.max_memory_allocated() / 2 ** 30
    loss = chunked_kl_loss(h, W0, b0, t_lg, chunk=64, temperature=1.0)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    dt = time.time() - t0
    print(f"[chunk=64] 教师@{tag:3s}  峰值={peak:6.2f} GiB (前向相位 {peak_fw:5.2f})  loss={loss.item():.6f}  {dt:4.1f}s", flush=True)
    return {'peak_gib': round(peak, 2), 'peak_fw_gib': round(peak_fw, 2), 'loss': loss.item(), 'sec': round(dt, 1)}


res['teacher_gpu'] = run(T_LG_GPU, 'GPU')
res['teacher_cpu'] = run(T_LG_CPU, 'CPU')

# 对照：全量路径峰值（教师 GPU）
model.zero_grad(set_to_none=True)
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
s_lg = model.forward(x)[0]
lA = full_kl_loss(s_lg.float(), T_LG_GPU, 1.0)
lA.backward()
torch.cuda.synchronize()
print(f"[全量    ] 峰值={torch.cuda.max_memory_allocated()/2**30:6.2f} GiB  loss={lA.item():.6f}  {time.time()-t0:4.1f}s")
res['full'] = {'peak_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), 'loss': lA.item()}

print('\n' + json.dumps(res, ensure_ascii=False, indent=1))
