"""FlashAttention 可用性门槛探测：head_dim / seqlen / num_heads / 是否 q is k / scale。

目的：搞清为什么 TF（head_dim=32）能命中 flash，而我们（head_dim=128）不能。
用法: python benchmarks/probe_flash.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

print('torch', torch.__version__, '|', torch.cuda.get_device_name(0),
      '| cc', torch.cuda.get_device_capability(0))
print('flash_sdp_enabled =', torch.backends.cuda.flash_sdp_enabled(),
      '| mem_efficient =', torch.backends.cuda.mem_efficient_sdp_enabled(),
      '| math =', torch.backends.cuda.math_sdp_enabled())
print()


def go(q, k, v, be, scale=1.0, causal=True):
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        return 'OK'
    except Exception:
        return 'NO'


print(f"{'head_dim':>9}{'seqlen':>8}{'heads':>7}{'q_is_k':>8}{'scale':>8}   FLASH  EFFIC")
for hd in (32, 64, 128, 256):
    for sl in (256, 1024):
        for nh in (1, 8):
            for same in (True, False):
                for scale in (1.0, hd ** -0.5):
                    B = 8
                    q = torch.randn(B, nh, sl, hd, device='cuda', dtype=torch.bfloat16)
                    k = q if same else torch.randn(B, nh, sl, hd, device='cuda', dtype=torch.bfloat16)
                    v = torch.randn(B, nh, sl, hd, device='cuda', dtype=torch.bfloat16)
                    f = go(q, k, v, SDPBackend.FLASH_ATTENTION, scale)
                    e = go(q, k, v, SDPBackend.EFFICIENT_ATTENTION, scale)
                    print(f'{hd:>9}{sl:>8}{nh:>7}{str(same):>8}{scale:>8.4f}   {f:>5}  {e:>5}')
                    del q, k, v
                    torch.cuda.empty_cache()

# 大 batch（我们 OPT6 的真实形状：nf=512）
print()
print('--- 真实形状（batch 维 = B*nh*nch） ---')
for nf, sl, hd in ((512, 256, 128), (2, 8192, 32), (16, 8192, 32), (256, 1024, 128)):
    q = torch.randn(nf, 1, sl, hd, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(nf, 1, sl, hd, device='cuda', dtype=torch.bfloat16)
    f = go(q, q, v, SDPBackend.FLASH_ATTENTION, 1.0)
    e = go(q, q, v, SDPBackend.EFFICIENT_ATTENTION, 1.0)
    print(f'  nf={nf:<6} sl={sl:<6} hd={hd:<5}  FLASH:{f}  EFFIC:{e}')
    del q, v; torch.cuda.empty_cache()
