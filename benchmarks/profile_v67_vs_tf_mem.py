"""v6.7 vs TF 显存构成拆解 —— 找出 v6.7 比 TF 多占的 ~4.2GiB 花在哪。

方法：分阶段测峰值显存（权重 / 前向激活 / 反向梯度 / 优化器），
     并用 memory_summary 打印大块分配，逐项归因。
"""
import sys, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
torch.manual_seed(0)
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
torch.backends.cuda.matmul.allow_tf32 = True

GiB = 2 ** 30


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def build(arch):
    if arch == 'v6.7':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
        return M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT,
                 W=W, read_mode='raw').to(DEV)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T).to(DEV)
    raise ValueError(arch)


x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head_small = torch.nn.Linear(D, 4096, bias=False).to(DEV)

for arch in ('v6.7', 'tf'):
    print('=' * 84)
    print(f'{arch}   T={T} W={W} batch={B}  D={D} nh={NH} n_layer={NLAYER} mlp_mult={MLP_MULT}')
    print('=' * 84)
    reset()
    m = build(arch)
    nparam = sum(p.numel() for p in m.parameters())
    base = torch.cuda.memory_allocated() / GiB
    print(f'  ① 模型权重+缓冲已分配        {base:7.3f} GiB   ({nparam:,} 参数)')
    if arch == 'v6.7':
        # 看有没有持久张量（memories / 位置编码 / slot 表）
        for name, buf in list(m.named_buffers())[:12]:
            print(f'       buffer {name:38s} {buf.numel()*buf.element_size()/GiB:7.4f} GiB  {tuple(buf.shape)}')

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    # ② 前向
    reset()
    o = m.forward_hidden(x)
    fwd_act = torch.cuda.max_memory_allocated() / GiB
    hidden_gib = o.numel() * o.element_size() / GiB
    print(f'  ② 前向峰值                   {fwd_act:7.3f} GiB   (hidden 输出 {tuple(o.shape)} = {hidden_gib:.3f} GiB)')
    del o

    # ③ fwd+bwd
    reset()
    o = m.forward_hidden(x)
    lg = o.view(B * T, D) @ head_small.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    lv = loss.item()
    loss.backward()
    bwd_act = torch.cuda.max_memory_allocated() / GiB
    print(f'  ③ +反向峰值                  {bwd_act:7.3f} GiB   (loss={lv:.1f})')

    # ④ +optim
    opt.step(); opt.zero_grad(set_to_none=True)
    opt_act = torch.cuda.max_memory_allocated() / GiB
    print(f'  ④ +优化器峰值                {opt_act:7.3f} GiB')
    print(f'  ⇒ 激活+梯度增量 = {opt_act-base:7.3f} GiB')
    print()
    print('  大块分配 top:')
    for stat in sorted([s for s in torch.cuda.memory_stats().get('allocation_sizes', [])], reverse=True)[:6]:
        pass
    segs = torch.cuda.memory_snapshot()
    big = []
    for s in segs:
        for blk in s.get('blocks', []):
            if blk.get('state') == 'active_allocated' and blk['size'] > 50 * 1024 * 1024:
                big.append(blk['size'] / GiB)
    for sz in sorted(big, reverse=True)[:8]:
        print(f'      {sz:7.3f} GiB')
    print()
    del m, opt, o, lg, loss
    reset()
