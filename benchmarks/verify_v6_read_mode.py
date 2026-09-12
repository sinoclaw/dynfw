"""验证 v6 加 read_mode 开关后的三项：
  ① softmax 默认路径与改前【逐位一致】（保 v6 现有读数不作废）
  ② raw 路径因果正确（长度依赖探针 maxdiff=0）
  ③ softmax vs raw 确实有差异（开关真生效）
用法: python verify_v6_read_mode.py save|check
"""
import hashlib
import json
import sys

import torch

from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM

DEV = 'cuda'
KW = dict(D=128, nh=16, vocab=151936, n_layer=2, steps=1, mlp_mult=64, W=64)
BASE = '/data/dynfw/results/v6_readmode_baseline.json'


def build(read_mode=None):
    torch.manual_seed(1234)
    if read_mode is None:
        return BDHBlockFWCycleLM(**KW).to(DEV).eval()
    try:
        return BDHBlockFWCycleLM(read_mode=read_mode, **KW).to(DEV).eval()
    except TypeError:
        print(f'  (模型不支持 read_mode={read_mode}，回退默认)')
        return BDHBlockFWCycleLM(**KW).to(DEV).eval()


def fp(m):
    """指纹：固定输入的前向 logits 摘要"""
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, 64), device=DEV)
    with torch.no_grad():
        lg = m(x)[0]
    return {'sum': float(lg.sum()), 'absmax': float(lg.abs().max()),
            'sample': [float(v) for v in lg[0, :2, 0].tolist()]}


def causal(m, POS=48):
    torch.manual_seed(7)
    x = torch.randint(0, 151936, (1, POS + 2), device=DEV)
    with torch.no_grad():
        a = m(x[:, :POS + 1])[0]
        b = m(x[:, :POS + 2])[0]
    return (a[0, POS] - b[0, POS]).abs().max().item()


mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
m_def = build()
print(f"[默认路径] sum={fp(m_def)['sum']:.8f} absmax={fp(m_def)['absmax']:.8f}")
print(f"[默认路径] 因果探针 maxdiff={causal(m_def):.3e}")

try:
    m_raw = build('raw')
    f_raw = fp(m_raw)
    print(f"[raw 路径] sum={f_raw['sum']:.8f} absmax={f_raw['absmax']:.8f}")
    print(f"[raw 路径] 因果探针 maxdiff={causal(m_raw):.3e} "
          f"({'CAUSAL OK' if causal(m_raw) < 1e-5 else 'LEAK!!'})")
    d = abs(f_raw['sum'] - fp(m_def)['sum'])
    print(f"[差异] softmax vs raw 的 logits sum 差 = {d:.6f} "
          f"({'开关生效 ✓' if d > 1e-6 else '⚠️ 无差异（开关没接上？）'})")
except Exception as e:
    print('raw 路径失败:', type(e).__name__, e)

# 新旧指纹比对
if mode == 'save':
    json.dump(fp(m_def), open(BASE, 'w'))
    print('baseline 已保存（改前 softmax 指纹）')
else:
    try:
        b = json.load(open(BASE))
        n = fp(m_def)
        same = (b['sum'] == n['sum'] and b['absmax'] == n['absmax'] and b['sample'] == n['sample'])
        print('① softmax 默认路径不变量:', 'PASS（逐位一致）' if same else 'FAIL —— 默认行为被改动！')
    except FileNotFoundError:
        print('无 baseline 可比')
