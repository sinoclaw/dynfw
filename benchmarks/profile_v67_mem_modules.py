"""逐子模块定位 v6.7 前向显存峰值（vs TF 同位置）。"""
import sys, gc
import torch

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


x = torch.randint(0, 1000, (B, T), device=DEV)

# ---------- v6.7 ----------
from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
from dynfw.models.fused_fw_gdn_fla import GDNFastAttnFLA

m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                  W=W, read_mode='raw').to(DEV)
print(f'v6.7 参数量 {sum(p.numel() for p in m.parameters()):,}')
print(f'  N (mlp internal) = {MLP_MULT*D//NH}   W={W}  T={T}  n_layer={NLAYER}')
print()

# 逐模块注册 hook 记录该模块【输出】的字节数（前向中一次性物化的张量）
sizes = []
def hook(name):
    def fn(mod, inp, out):
        tot = 0
        for o in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(o):
                tot += o.numel() * o.element_size()
        if tot > 20 * 1024 * 1024:
            sizes.append((name, mod.__class__.__name__, tot / GiB))
    return fn

for n, mod in m.named_modules():
    if n:
        mod.register_forward_hook(hook(n))

reset()
with torch.no_grad():
    o = m.forward_hidden(x)
peak = torch.cuda.max_memory_allocated() / GiB
print(f'v6.7 前向峰值 = {peak:.3f} GiB')
print('  输出 >20MB 的模块（一次性物化的大张量）:')
for n, cls, sz in sorted(sizes, key=lambda t: -t[2])[:14]:
    print(f'    {sz:7.4f} GiB  {n:44s} {cls}')
del m
reset()

# ---------- TF ----------
from dynfw.models.transformer import TF_sdpa
m2 = TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
sizes2 = []
for n, mod in m2.named_modules():
    if n:
        mod.register_forward_hook(hook(n))
print()
print(f'TF 参数量 {sum(p.numel() for p in m2.parameters()):,}')
reset()
with torch.no_grad():
    o2 = m2.forward_hidden(x)
peak2 = torch.cuda.max_memory_allocated() / GiB
print(f'TF 前向峰值 = {peak2:.3f} GiB')
print('  输出 >20MB 的模块:')
for n, cls, sz in sorted(sizes2, key=lambda t: -t[2])[:10]:
    print(f'    {sz:7.4f} GiB  {n:44s} {cls}')
