"""诊断：长 T 下 DynFW(v6-opt5) vs TF —— per-position KL 分布。

问题：输在「长依赖（序列后半段）」还是「整体拟合」？
做法：加载两个已训练 checkpoint，在同一批数据（训练用的前 5 块）+ 同一教师 logits 上
      逐位置算 KL，按位置分桶（1/4 段）比较。
判读：若 v6 后半段相对 TF 明显更差 ⇒ 长依赖是短板；
      若各段差距均匀 ⇒ 整体拟合/优化效率差异。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM     # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw       # noqa: E402
from dynfw.models.transformer import TF_sdpa                     # noqa: E402

R = '/data/dynfw/results'
VOCAB, D, NH, NL, MM = 151936, 128, 16, 2, 64
T = 8192
NB = 5          # 训练用的块数
DEV = 'cuda'


def load_v6(ckpt):
    torch.manual_seed(0)
    m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=MM, W=64,
                          read_mode='raw')
    m = to_opt5_raw(m, strict_bf16=True, bf16_prefix=False)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd if not isinstance(sd, dict) or 'model' not in sd else sd['model'])
    return m.to(DEV).eval()


def load_tf(ckpt):
    torch.manual_seed(0)
    m = TF_sdpa(D=D, nh=NH, n_layer=NL, vocab=VOCAB, maxT=T)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd if not isinstance(sd, dict) or 'model' not in sd else sd['model'])
    return m.to(DEV).eval()


# 数据：与训练一致的前 NB 块
arr = np.fromfile('/data/dynfw/data/tinystories_qwen.bin', dtype=np.uint32)[:NB * T].astype(np.int64)
ids = torch.from_numpy(arr).view(NB, T)

# 教师 logits（复用缓存）
tj = os.path.join(R, 'lt8192opt5_fw_s0', 'teacher_logits', 'teacher_logits.npy')
assert os.path.exists(tj), f'缺教师 logits: {tj}'
t_lg = np.load(tj, mmap_mode='r')
print(f'教师 logits: {t_lg.shape} {t_lg.dtype}')

CH = 1024


def per_pos_kl(m):
    """两侧共用同一接口：forward_hidden() 返回 lm_head 之前的 hidden，head_params() 给出 (W,b)。"""
    out = torch.zeros(NB, T)
    W_, b_ = m.head_params()
    Wf = W_.float()
    bf = b_.float() if b_ is not None else None
    for b in range(NB):
        x = ids[b:b + 1].to(DEV)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            h = m.forward_hidden(x)
        # 教师 logits 逐 chunk 从 mmap 读入：避免一次性物化 (T,V) fp32（5 GB）导致 OOM
        for i in range(0, T, CH):
            hi = h[:, i:i + CH]
            z = hi.float() @ Wf.t() + (bf if bf is not None else 0)
            tnp = np.asarray(t_lg[b, i:i + CH], dtype=np.float32)       # (chunk,V) 来自 mmap
            tt = torch.from_numpy(tnp).to(DEV)
            ls = F.log_softmax(z, dim=-1)
            p = F.softmax(tt, dim=-1)
            kl = (p * (torch.log(p + 1e-12) - ls)).sum(-1)   # (1,chunk)
            out[b, i:i + CH] = kl[0].float().cpu()
            del z, ls, p, kl, tt, tnp, hi
            torch.cuda.empty_cache()
        del h, x
    return out


m_fw = load_v6(os.path.join(R, 'lt8192opt5_fw_s0', 'checkpoint', 'student.pt'))
m_tf = load_tf(os.path.join(R, 'lt8192opt5_tf_s0', 'checkpoint', 'student.pt'))

print('计算 per-position KL ...')
kl_fw = per_pos_kl(m_fw)
print('  v6 done')
kl_tf = per_pos_kl(m_tf)
print('  tf done')

print(f'\n总 KL:  v6={kl_fw.mean():.4f}  tf={kl_tf.mean():.4f}  (差 {kl_fw.mean()-kl_tf.mean():+.4f})')
print(f'\n按位置分 8 段（每段 {T//8} token）的平均 per-token KL:')
print(f'{"段":>3s} {"位置范围":>14s} {"v6":>10s} {"tf":>10s} {"v6/tf":>8s}')
for i in range(8):
    a, b = i * T // 8, (i + 1) * T // 8
    v = kl_fw[:, a:b].mean().item()
    t = kl_tf[:, a:b].mean().item()
    print(f'{i+1:3d} {f"[{a},{b})":>14s} {v:10.4f} {t:10.4f} {v/t:8.3f}')

print(f'\n前 1/4 段:  v6={kl_fw[:, :T//4].mean():.4f}  tf={kl_tf[:, :T//4].mean():.4f}  比={kl_fw[:, :T//4].mean()/kl_tf[:, :T//4].mean():.3f}')
print(f'后 1/4 段:  v6={kl_fw[:, -T//4:].mean():.4f}  tf={kl_tf[:, -T//4:].mean():.4f}  比={kl_fw[:, -T//4:].mean()/kl_tf[:, -T//4:].mean():.3f}')
print('\n判读: 末段比值 >> 首段比值 ⇒ 长依赖是短板；各段比值接近 ⇒ 整体拟合差异。')
