"""收尾轮：① 性能变体（bf16 softmax / max-autotune）② J3 短训等价性（欠账补上）。

用法: PYTHONPATH=/data/dynfw python benchmarks/final_round.py --part all
"""
import sys, time, argparse
import numpy as np
import torch

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import to_opt5
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144
TRAIN_BIN = '/data/corpus/tinystories/train.bin'
VAL_BIN = '/data/corpus/tinystories/valid.bin'


def build_base(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL, steps=1, mlp_mult=4, W=W)


def build_tf():
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=BIG_T)


def mk_base():
    return build_base()


def mk_tfc():
    return torch.compile(build_tf())


def mk_o5c():
    return torch.compile(to_opt5(build_base(), True, False, False))


def mk_o5bs():
    return torch.compile(to_opt5(build_base(), True, False, True))


def mk_o5ma():
    return torch.compile(to_opt5(build_base(), True, False, False),
                         mode='max-autotune-no-cudagraphs')


def run_step(m, opt, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return loss


def bench_median(m, T, B=2, rounds=3, iters=3, warmup=3):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    y = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(warmup):
        run_step(m, opt, x, y)
    torch.cuda.synchronize()
    s = []
    for _ in range(rounds):
        t0 = time.time()
        for _ in range(iters):
            run_step(m, opt, x, y)
        torch.cuda.synchronize()
        s.append(1000 * (time.time() - t0) / iters)
    s.sort()
    return s[len(s) // 2], s[0]


def part1():
    Ts = (8192, 16384, 32768)
    COLS = [('TF+comp', mk_tfc), ('OPT5+comp', mk_o5c),
            ('OPT5bs+comp', mk_o5bs), ('OPT5ma+comp', mk_o5ma)]
    print('=' * 92)
    print('Part 1 性能变体（中位/最小 ms·step，batch=2）')
    print('=' * 92)
    print(f"{'T':>7}" + ''.join(f'{n:>15}' for n, _ in COLS))
    res = {}
    for T in Ts:
        row = f'{T:>7}'
        for name, mk in COLS:
            try:
                m = mk().cuda()
                med, mn = bench_median(m, T)
                res[(name, T)] = med
                row += f"{f'{med:.1f}ms':>15}"
                del m; torch.cuda.empty_cache()
            except Exception as e:
                row += f"{'ERR':>15}"
                torch.cuda.empty_cache()
                print(f'   !! {name} T={T}: {type(e).__name__} {str(e)[:60]}')
        print(row)
    print()
    for T in Ts:
        tf = res.get(('TF+comp', T))
        line = f'  T={T:<6}'
        for n in ('OPT5+comp', 'OPT5bs+comp', 'OPT5ma+comp'):
            v = res.get((n, T))
            if tf and v:
                line += f'  {n}/TF={tf/v:5.2f}x'
        print(line)


def load_bins():
    tr = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
    va = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
    return tr, va


def make_getter(arr, T, B, rng):
    def get():
        ix = rng.integers(0, len(arr) - T - 1, size=B)
        xs = np.stack([arr[i:i + T].astype(np.int64) for i in ix])
        ys = np.stack([arr[i + 1:i + 1 + T].astype(np.int64) for i in ix])
        return torch.from_numpy(xs).cuda(), torch.from_numpy(ys).cuda()
    return get


def part2(steps=300, B=16, T=1024):
    tr, va = load_bins()
    print()
    print('=' * 92)
    print(f'Part 2 J3 短训等价性（TinyStories, {steps} 步, batch={B}, T={T}, 同 seed）')
    print('=' * 92)
    arms = [('基线 v6', mk_base), ('OPT5+comp', mk_o5c), ('TF+comp', mk_tfc)]
    for name, mk in arms:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        get = make_getter(tr, T, B, rng)
        m = mk().cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        curve = []
        t0 = time.time()
        for s in range(1, steps + 1):
            x, y = get()
            loss = run_step(m, opt, x, y)
            if s % 50 == 0:
                curve.append((s, float(loss)))
        wall = time.time() - t0
        # 固定验证批
        rng2 = np.random.default_rng(999)
        gv = make_getter(va, T, 8, rng2)
        m.eval()
        vl = []
        with torch.no_grad():
            for _ in range(4):
                xv, yv = gv()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, lv = m(xv, yv)
                vl.append(float(lv))
        vmean = sum(vl) / len(vl)
        print(f'  {name:<12} 末步 train={curve[-1][1]:.4f}  验证={vmean:.4f}  '
              f'墙钟={wall:.0f}s  曲线=' + ' '.join(f'{s}:{l:.3f}' for s, l in curve))
        del m, opt; torch.cuda.empty_cache()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all')
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    if a.part in ('all', '1'):
        part1()
    if a.part in ('all', '2'):
        part2(steps=a.steps)
