"""峰值溯源：用 _record_memory_history 抓每一次分配的调用栈，
数清 5.13 GiB 峰值到底由谁构成。

判据（跑前锁死）：
  J1 定位：给出峰值时刻占用最大的 8 类分配及其调用栈
  J2 归因：标注每类属于「权重 / 激活 / 梯度 / FLA kernel 工作区 / 其他」
  J3 可操作性：指出哪些是可优化的（能改代码动它），哪些是结构必须
"""
import sys, pickle, collections
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
GiB = 2 ** 30
MB = 2 ** 20
torch.backends.cuda.matmul.allow_tf32 = True

from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)

import types, json, os

def probe(tag, **kw):
    torch.manual_seed(0)
    m = BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                      mlp_mult=MLP_MULT, W=W, read_mode='raw', **kw).to(DEV)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def one():
        o = m.forward_hidden(x)
        lg = o.view(B * T, D) @ head.weight.T
        loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(2):
        one()
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=200000, context='all')
    lv = one()
    torch.cuda.synchronize()
    path = f'/tmp/memsnap_{tag}.pickle'
    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(enabled=None)

    pk = torch.cuda.max_memory_allocated() / GiB
    print(f"[{tag}] peak={pk:.3f} GiB  loss={lv:.6f}  snapshot={path}")

    del m, opt
    torch.cuda.empty_cache()
    return path, pk


def analyze(path, tag):
    with open(path, 'rb') as f:
        snap = pickle.load(f)

    # 收集所有分配事件的 (size, frames)，按 frames 聚合
    agg = collections.defaultdict(lambda: [0, 0])   # frames_key -> [total_bytes, count]
    for act in snap.get('device_traces', []):
        for a in act:
            if a.get('action') != 'alloc':
                continue
            sz = a.get('size') or 0
            frames = a.get('frames') or []
            # 取调用栈里第一帧用户代码（跳过 torch 内部）
            key = None
            for fr in frames:
                fn = fr.get('filename') or ''
                if '/data/dynfw/' in fn or 'dynfw' in fn:
                    key = f"{os.path.basename(fn)}:{fr.get('line')} {fr.get('name')}"
                    break
            if key is None and frames:
                fr = frames[0]
                key = f"{os.path.basename(fr.get('filename') or '?')}:{fr.get('line')} {fr.get('name')}"
            if key is None:
                key = '(unknown)'
            agg[key][0] += sz
            agg[key][1] += 1

    print()
    print("=" * 104)
    print(f"峰值溯源 [{tag}]：按调用栈聚合的总分配字节（Top 12）")
    print("=" * 104)
    tot = sum(v[0] for v in agg.values())
    print(f"  总分配量 {tot/GiB:.2f} GiB（含全部临时；峰值是其中同时存活的部分）")
    print()
    for k, (byt, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {byt/GiB:8.3f} GiB  ×{n:<6d}  {k[:96]}")
    print()


p0, pk0 = probe('base')
analyze(p0, 'base')

p1, pk1 = probe('ckpt', grad_ckpt=True)
analyze(p1, 'ckpt')
print(f"对比：base peak={pk0:.3f} GiB → ckpt peak={pk1:.3f} GiB")
