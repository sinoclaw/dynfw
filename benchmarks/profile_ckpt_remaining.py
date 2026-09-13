"""深挖：grad_ckpt 后仍占 3.69GiB 的是什么？

方法：在 grad_ckpt=True 下逐阶段测峰值，并用 memory_snapshot 看哪类张量常驻。
"""
import sys, gc, collections
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
torch.backends.cuda.matmul.allow_tf32 = True

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


for ckpt in (False, True):
    print("=" * 92)
    print(f"grad_ckpt = {ckpt}")
    print("=" * 92)
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw', grad_ckpt=ckpt).to(DEV)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    reset()
    print(f"  ① 权重+缓冲          {torch.cuda.memory_allocated()/GiB:7.3f} GiB")

    # 单层前向峰值（no_grad）
    reset()
    with torch.no_grad():
        o = m.forward_hidden(x)
    print(f"  ② 前向峰值(no_grad)  {torch.cuda.max_memory_allocated()/GiB:7.3f} GiB")
    del o

    # fwd+bwd 峰值
    reset()
    o = m.forward_hidden(x)
    lg = o.view(B * T, D) @ head.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    loss.backward()
    pk = torch.cuda.max_memory_allocated() / GiB
    print(f"  ③ fwd+bwd 峰值       {pk:7.3f} GiB")
    opt.zero_grad(set_to_none=True)

    # 看常驻的大块（backward 后仍存活 = 常驻激活/梯度）
    segs = torch.cuda.memory_snapshot()
    big = []
    for s in segs:
        for blk in s.get('blocks', []):
            if blk.get('state') == 'active_allocated' and blk['size'] > 20 * 1024 * 1024:
                big.append(blk['size'] / GiB)
    print(f"  ④ backward 后常驻 >20MB 块: {len(big)} 个, 合计 {sum(big):.3f} GiB")
    for sz in sorted(big, reverse=True)[:6]:
        print(f"       {sz:7.3f} GiB")

    # 按形状统计所有活跃张量
    print("  ⑤ 各形状激活的元素数（按形状聚合）")
    shapes = collections.defaultdict(lambda: [0, 0])
    def walk(t):
        if torch.is_tensor(t):
            shapes[tuple(t.shape)][0] += t.numel() * t.element_size()
            shapes[tuple(t.shape)][1] += 1
        elif isinstance(t, (list, tuple)):
            for u in t: walk(u)
    walk(list(m.state_dict().items()))
    for sh, (byt, n) in sorted(shapes.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"       {byt/GiB:7.4f} GiB  ×{n:3d}  {sh}")

    del m, opt
    reset()
    print()
