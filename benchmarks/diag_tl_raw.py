"""诊断：FWAttentionTLRaw 接进模型后为何失败（最小复现）。"""
import sys

import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM          # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_tl_raw              # noqa: E402

torch.manual_seed(0)
D, NH, NL, MM, W = 128, 16, 2, 64, 64
VOCAB = 151936

for T in (256, 1024):
    print(f'\n===== T={T} (W={W}, nch={T // W}) =====')
    m = to_tl_raw(BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1,
                                    mlp_mult=MM, W=W, read_mode='raw')).cuda().eval()
    print('  attn 类:', type(m.blocks[0].attn).__name__)
    x = torch.randint(0, VOCAB, (1, T), device='cuda')
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = m(x)
        torch.cuda.synchronize()
        print(f'  ✅ 前向 OK  logits shape={tuple(lg.shape)}  '
              f'|logits|max={lg.abs().max().item():.4f}')
    except Exception as e:
        import traceback
        print(f'  ❌ 失败: {type(e).__name__}: {str(e)[:600]}')
        tb = traceback.format_exc().splitlines()
        print('  --- 关键行 ---')
        for line in tb[-14:]:
            print('   ', line[:160])
