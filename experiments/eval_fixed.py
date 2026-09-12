"""固定口径评测：给 val loss 一个确定的抽样 seed，并量化抽样噪声。

背景漏洞：train_talk.py 的 eval_loss 用 torch.randint 抽 val 批次、没固定 seed，
所以同一个 ckpt 的 val_loss 每次都不一样（实测同一 TF ckpt 1.5002 vs 1.4882，差 0.012）。
在「要判断 0.075 nats 的 N/D 效应是真是假」之前，必须先把这个噪声测出来。

用法:
  # 单个 ckpt，固定 seed 123（= compare_talk.py 的口径）
  python eval_fixed.py --ckpt results/talk2_tf/ckpt.pt --arch tf --eval-seed 123

  # 噪声扫描：同一 ckpt 换 10 个 eval seed，看 val_loss 的均值/标准差
  python eval_fixed.py --ckpt results/talk2_tf/ckpt.pt --arch tf --noise-scan
"""
import argparse, json, math, sys
import numpy as np, torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.transformer import TF_sdpa

DATA = '/data/corpus/tinystories'

ARCH_CFG = {
    'v6':  dict(D=128, nh=8,  mm=16, W=256),
    'v6w': dict(D=256, nh=8,  mm=4,  W=256),
}


def build(arch, vocab=50257, D=None, nh=None, mm=None, n_layer=6, W=256):
    if arch in ARCH_CFG:
        c = ARCH_CFG[arch]
        return BDHBlockFWCycleLM(D=c['D'], nh=c['nh'], vocab=vocab, n_layer=n_layer,
                                 steps=1, mlp_mult=c['mm'], W=c['W'])
    if arch == 'cfg':
        return BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer,
                                 steps=1, mlp_mult=mm, W=W)
    if arch == 'tf':
        return TF_sdpa(D=256, nh=8, n_layer=6, vocab=vocab)
    raise ValueError(arch)


def batches(data, B, T, n, gen):
    """固定 seed 抽 n 个批次（与 compare_talk.py 同款：随机偏移 + 固定生成器）"""
    for _ in range(n):
        ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
        yield x.cuda(), y.cuda()


@torch.no_grad()
def eval_fixed(model, va, B, T, seed, iters=40):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ls = []
    for x, y in batches(va, B, T, iters, gen):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        ls.append(loss.item())
    return sum(ls) / len(ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', required=True, choices=['v6', 'v6w', 'tf', 'cfg'])
    ap.add_argument('--D', type=int); ap.add_argument('--nh', type=int); ap.add_argument('--mm', type=int)
    ap.add_argument('--eval-seed', type=int, default=123)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--noise-scan', action='store_true', help='换 10 个 eval seed 测抽样噪声')
    a = ap.parse_args()

    va = np.memmap(f'{DATA}/valid.bin', dtype=np.uint16, mode='r')
    m = build(a.arch, D=a.D, nh=a.nh, mm=a.mm).cuda()
    ck = torch.load(a.ckpt, map_location='cuda')
    m.load_state_dict(ck['model'])
    m.eval()

    pt = ck.get('params', {})
    print(f'=== {a.ckpt}  arch={a.arch}  ckpt记录val_loss={ck.get("val_loss")} ===')
    print(f'    params: total={pt.get("total")}  struct={pt.get("struct")}')
    print(f'    抽样口径: {a.iters} iters x B{a.batch} x T{a.T} = {a.iters*a.batch*a.T:,} tokens '
          f'(valid.bin 共 {len(va):,} tokens = {a.iters*a.batch*a.T/len(va):.1%})')

    if a.noise_scan:
        seeds = [123] + list(range(1, 10))
        res = {}
        for s in seeds:
            v = eval_fixed(m, va, a.batch, a.T, s, a.iters)
            res[s] = v
            print(f'    eval_seed={s:<4} val_loss={v:.6f}')
        vals = np.array(list(res.values()))
        print(f'  --> 均值 {vals.mean():.6f}  标准差 {vals.std(ddof=1):.6f}  '
              f'极差 {vals.max()-vals.min():.6f}')
        print(f'  --> 单次抽样的 1σ 噪声 ≈ {vals.std(ddof=1):.4f} nats')
        json.dump({'ckpt': a.ckpt, 'arch': a.arch, 'per_seed': res,
                   'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)),
                   'range': float(vals.max() - vals.min())},
                  open(f'/data/logs/evalnoise_{a.arch}.json', 'w'), indent=1)
    else:
        v = eval_fixed(m, va, a.batch, a.T, a.eval_seed, a.iters)
        print(f'  val_loss (eval_seed={a.eval_seed}) = {v:.6f}   ppl={math.exp(v):.4f}')
        json.dump({'ckpt': a.ckpt, 'arch': a.arch, 'eval_seed': a.eval_seed,
                   'val_loss': v, 'val_ppl': math.exp(v)},
                  open(f'/data/logs/evalfixed_{a.arch}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
