"""验证 fused_fw_full.py 的 softmax 因果修复。
  A) raw 路径不变量：修复前后 logits 必须逐位一致（保批1 读数 93.23 不作废）
  B) softmax 路径因果性：修复前 LEAK → 修复后 CAUSAL OK
用法: python verify_full_softmax_fix.py save|check
"""
import json
import sys

import torch

from dynfw.models.fused_fw_full import FusedFWFull

DEV = 'cuda'


def build(use_softmax, n_layer=3):
    torch.manual_seed(1234)
    return FusedFWFull(D=64, N=512, k=16, nh=4, mlp_mult=16, vocab=1000,
                       use_ffn=False, n_layer=n_layer, use_softmax=use_softmax).to(DEV).eval()


def raw_logits(m):
    torch.manual_seed(7)
    x = torch.randint(0, 1000, (1, 64), device=DEV)
    with torch.no_grad():
        return m(x)[0]


def causal_maxdiff(m, POS=48):
    torch.manual_seed(7)
    x = torch.randint(0, 1000, (1, POS + 2), device=DEV)
    with torch.no_grad():
        a = m(x[:, :POS + 1])[0]
        b = m(x[:, :POS + 2])[0]
    return (a[0, POS] - b[0, POS]).abs().max().item()


mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
mraw = build(False)
msm = build(True)

lg = raw_logits(mraw)
sig = {'sum': float(lg.sum()), 'absmax': float(lg.abs().max()),
       'sample': [float(v) for v in lg[0, :3, 0].tolist()]}
print(f"raw_sum={sig['sum']:.8f}  raw_absmax={sig['absmax']:.8f}")
print(f"raw_sample={sig['sample']}")

d = causal_maxdiff(msm)
print(f"softmax 路径 maxdiff={d:.6e} -> {'CAUSAL OK' if d < 1e-5 else 'LEAK'}")

# 附带：softmax 路径是否出现 NaN（第一行无可见位置时的边界）
ssm = None
try:
    torch.manual_seed(7)
    x = torch.randint(0, 1000, (1, 32), device=DEV)
    with torch.no_grad():
        o = msm(x)[0]
    print(f"softmax 输出含 NaN: {bool(torch.isnan(o).any())}")
except Exception as e:
    print('softmax 前向异常:', type(e).__name__, e)

BASE = '/data/dynfw/results/full_raw_baseline.json'
if mode == 'save':
    json.dump(sig, open(BASE, 'w'))
    print('baseline 已保存')
else:
    try:
        base = json.load(open(BASE))
        same = (base['sum'] == sig['sum'] and base['absmax'] == sig['absmax']
                and base['sample'] == sig['sample'])
        print('raw 路径不变量:', 'PASS（逐位一致）' if same else 'FAIL —— raw 行为被改动！')
    except FileNotFoundError:
        print('无 baseline 可比对')
