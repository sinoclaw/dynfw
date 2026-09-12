"""op 级 profile：定位 v6.7(377.5ms) 比 v6+opt5(91.2ms) 多花的 286ms。

已知（纯计算口径, T=8192, W=64, fp32）:
  v6+opt5  前向 28.9ms   fwd+bwd  91.2ms   ← opt5 反向仅 ~62ms
  v6.7     前向 27.7ms   fwd+bwd 377.5ms   ← 前向持平，反向 ~350ms（12.6× 前向）
⟹ 前向没问题，钱花在【反向】。本脚本按 op 排名，区分 fwd / bwd。
"""
import torch
import torch.nn.functional as F

DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, MLP_MULT, VOCAB = 128, 16, 64, 151936

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM


def build_v67():
    return BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=2, steps=1, mlp_mult=MLP_MULT,
                         W=W, read_mode='raw', gate_mode='token').to(DEV)


m = build_v67()
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)
x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 4096, (B, T), device=DEV)


def step():
    o = m.forward_hidden(x)
    lg = o.view(B * T, D) @ head.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    loss.backward()


# 预热（含 FLA 的 autotune / JIT）
for _ in range(4):
    m.zero_grad(set_to_none=True)
    step()
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile

m.zero_grad(set_to_none=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    step()
torch.cuda.synchronize()

# 只统计真正的 CUDA kernel / aten 算子；排除 autograd 包装层（它只是 CPU 侧调度，
# 计进来会重复计数并产生 >100% 的荒谬占比）
def is_real(k):
    if k.startswith('autograd::'):
        return False
    if k.endswith('Backward0') or k.endswith('Backward'):
        return False
    return True

print('=== v6.7 op 级排名（按 CUDA 总时间，Top 15）===')
all_evts = prof.key_averages()
step_ms = sum(e.device_time_total for e in all_evts
              if e.device_time_total > 0 and is_real(e.key)) / 1000
evts = [e for e in all_evts if e.device_time_total > 0 and is_real(e.key)]
evts.sort(key=lambda e: -e.device_time_total)
print(f'{"op":52s} {"CUDA总":>10s} {"次数":>6s} {"占步长":>8s}')
total_us = 0.0
for e in evts[:15]:
    total_us += e.device_time_total
    print(f'{e.key[:52]:52s} {e.device_time_total/1000:9.2f}ms {e.count:6d} '
          f'{e.device_time_total/1000/step_ms*100:7.1f}%')
print(f'{"[真算子合计]":52s} {step_ms:9.2f}ms')

print()
print('=== 按前缀归并（看是哪一类开销）===')
buckets = {}
for e in evts:
    k = e.key
    kl = k.lower()
    if 'simple_gla' in kl or 'fusedchunk' in kl or 'gla_fwd' in kl or 'gla_bwd' in kl or 'chunk_o' in kl or 'chunk_h' in kl or 'chunk_fwd' in kl or 'chunk_bwd' in kl or 'chunk_g' in kl or 'cumsum' in kl:
        b = 'FLA kernel 系'
    elif 'copy' in k.lower() or 'cat' in k.lower() or 'view' in k.lower() or 'permute' in k.lower():
        b = '内存搬运系'
    elif 'elementwise' in k.lower() or 'mul' in k.lower() or 'add' in k.lower() or 'sigmoid' in k.lower():
        b = '逐元素/激活系'
    elif 'gemm' in k.lower() or 'mm' in k.lower() or 'bmm' in k.lower() or 'einsum' in k.lower():
        b = '矩阵乘系'
    elif 'softmax' in k.lower() or 'cross_entropy' in k.lower() or 'loss' in k.lower():
        b = '损失系'
    elif 'ln' in k.lower() or 'norm' in k.lower():
        b = '归一化系'
    else:
        b = '其它'
    buckets[b] = buckets.get(b, 0) + e.device_time_total
for b, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f'  {b:16s} {v/1000:9.2f}ms  {v/1000/step_ms*100:5.1f}%')
print()
print('=== 说明 ===')
print('  若 FLA kernel 系占比高 ⇒ 优化 FLA 调用（chunk_size / 换 fused_recurrent / 关 autotune）')
print('  若内存搬运系占比高 ⇒ 去掉 permute/expand/contiguous 与 fp32 强转（改成 FLA 原生布局）')
