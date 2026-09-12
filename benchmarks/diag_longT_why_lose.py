"""诊断：长 T（8192）下 DynFW 为何落后 TF —— 数值健康度。

怀疑点：
  A) v6 跨块 fast-weight 无界累加 M += k⊗v（T=8192/W=64 → 128 chunks）⇒ ‖M‖ 可能失控
  B) v6.6 门控 M = α·M + k⊗v 本应控制它 —— 对比两者
  C) hidden / logits 量级
对照：tf（全局 softmax，天生有界）、v5（整段精确 O(T²)，无 fast-weight）
"""
import sys

import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM     # noqa: E402
from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM   # noqa: E402
from dynfw.models.transformer import TF_sdpa                     # noqa: E402

VOCAB, D, NH, NL, MM, W = 151936, 128, 16, 2, 64, 64
DEV = 'cuda'
Ts = [256, 1024, 4096, 8192]


def build(name):
    torch.manual_seed(0)
    if name == 'v6':
        return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=MM, W=W, read_mode='raw')
    if name == 'v6.6':
        return BDHBlockGDNCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=MM, W=W, read_mode='raw')
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=32768)


def attach_mem_probe(m, box):
    """钩住 attn.forward，记录 new_mem 的 |max|（兼容返回 tuple / 单值）。"""
    saved = []
    for blk in getattr(m, 'blocks', []):
        attn = getattr(blk, 'attn', None)
        if attn is None or not hasattr(attn, 'forward'):
            continue
        orig = attn.forward

        def mk(orig_fn):
            def wrapped(Q, K, V, memories=None, W=512, **kw):
                r = orig_fn(Q, K, V, memories=memories, W=W, **kw)
                if isinstance(r, tuple) and len(r) == 2 and torch.is_tensor(r[1]):
                    box[0] = max(box[0], float(r[1].abs().max()))
                return r
            return wrapped
        attn.forward = mk(orig)
        saved.append((attn, orig))
    return saved


print(f'{"arch":6s} {"T":>6s} {"|M|max":>13s} {"|hidden|max":>12s} {"|logits|max":>12s} {"loss":>10s}')
print('-' * 68)
for name in ('v6', 'v6.6', 'tf'):
    m = build(name).to(DEV).train()
    for T in Ts:
        box = [0.0]
        saved = attach_mem_probe(m, box)
        torch.manual_seed(7)
        x = torch.randint(0, VOCAB, (1, T), device=DEV)
        y = torch.randint(0, VOCAB, (1, T), device=DEV)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, loss = m(x, y)
            hid = m.forward_hidden(x) if hasattr(m, 'forward_hidden') else None
            hm = hid.abs().max().item() if hid is not None else float('nan')
            print(f'{name:6s} {T:6d} {box[0]:13.3f} {hm:12.3f} '
                  f'{lg.float().abs().max().item():12.3f} {loss.item():10.2f}', flush=True)
        except Exception as e:
            print(f'{name:6s} {T:6d}  FAILED: {type(e).__name__}: {str(e)[:60]}', flush=True)
        finally:
            for attn, orig in saved:
                attn.forward = orig
        del x, y
        torch.cuda.empty_cache()
    del m
    torch.cuda.empty_cache()
