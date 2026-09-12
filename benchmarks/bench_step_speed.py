"""干净测速：完整训练 step（forward+backward+optimizer）的墙钟，opt5_raw vs TF。

为什么需要单独测：
  收口实验的 wall 被「mmap 读 47GB 教师 logits」的 IO 主导（v6/tf 各 seed 间隔几乎相同），
  不能反映算力差异。此脚本在单进程内固定形态、预热后取中位，隔离 IO。

形态（与收口实验一致）：
  v6 : opt5_raw（交付形态，eager，不 compile）
  tf : SDPA 融合核
口径：T=8192 / B=1 / D=128 nh=16 L=2 mm=64 / bf16 autocast；预热 20 step，取 30 step 中位。
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM      # noqa: E402
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw        # noqa: E402
from dynfw.models.transformer import TF_sdpa                      # noqa: E402
from dynfw.training.chunked_kl import chunked_kl_loss             # noqa: E402

VOCAB, D, NH, NL, MM, W = 151936, 128, 16, 2, 64, 64
Ts = [int(x) for x in (sys.argv[1:] or ['8192'])]
DEV = 'cuda'
WARM, MEAS = 20, 30


def build(kind, T):
    torch.manual_seed(0)
    if kind == 'v6':
        m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=MM, W=W,
                              read_mode='raw')
        m = to_opt5_raw(m, strict_bf16=True, bf16_prefix=False)
    else:
        m = TF_sdpa(D=D, nh=NH, n_layer=NL, vocab=VOCAB, maxT=T)
    return m.to(DEV).train()


def one_step(m, opt, x, t_lg):
    """与收口实验同形态：forward_hidden + 分块 KL（不物化 BxTxV）。"""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = m.forward_hidden(x)
        Wh, bh = m.head_params()
        loss = chunked_kl_loss(h, Wh, bh, t_lg, chunk=2048)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss


def bench(kind, T):
    m = build(kind, T)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (1, T), device=DEV)
    # 假教师 logits 常驻显存：隔离 mmap IO（收口实验里 IO 淹没了算力差异）
    t_lg = torch.zeros(1, T, VOCAB, dtype=torch.float16, device=DEV)
    for _ in range(WARM):
        one_step(m, opt, x, t_lg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(MEAS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one_step(m, opt, x, t_lg)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    del m, opt, x, t_lg
    torch.cuda.empty_cache()
    return statistics.median(ts), min(ts)


print(f'{"T":>7s} {"v6(opt5_raw)":>14s} {"tf(SDPA)":>10s} {"tf/v6":>8s} {"谁快":>8s}')
print('-' * 54)
for T in Ts:
    mv, _ = bench('v6', T)
    mt, _ = bench('tf', T)
    who = 'v6' if mv < mt else 'tf'
    print(f'{T:7d} {mv:11.2f} ms {mt:7.2f} ms {mt/mv:8.3f} {who:>8s}', flush=True)
print('\n注：完整训练 step（fwd+bwd+opt），单进程固定形态，预热 20 后取 30 step 中位。')
