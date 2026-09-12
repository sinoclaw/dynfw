"""小模型「会说话」对比验证：并排跑我们的架构 vs 标准 Transformer。

用法:
  PYTHONPATH=/data/dynfw python experiments/compare_talk.py \
      --a results/talk2_v6/ckpt.pt:v6 --b results/talk2_tf/ckpt.pt:tf

输出：参数量 / val loss / top-1 / 分布外 loss / 贪心生成 / 采样生成（并排）
"""
import sys, os, argparse, math
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from tokenizers import Tokenizer

TRAIN_BIN = '/data/corpus/tinystories/train.bin'
VALID_BIN = '/data/corpus/tinystories/valid.bin'
TOK_PATH = '/data/models/gpt2-tok/tokenizer.json'
VOCAB = 50257

# 与 train_talk.py 保持一致的架构配置（同结构参数 4.72M、同深度 6 层）
ARCH_CFG = {
    'v6': dict(D=128, nh=8, n_layer=6, steps=1, mlp_mult=16, W=256),
    'v6w': dict(D=256, nh=8, n_layer=6, steps=1, mlp_mult=4, W=256),
    'tf': dict(D=256, nh=8, n_layer=6),
}

PROMPTS = [
    'Once upon a time, there was a little girl named Lily.',
    'One day, a small dog went to the park.',
    'Tom was hungry, so he',
    'The little cat saw a big red ball and',
]


def build(arch, cfg):
    if arch in ('v6', 'v6w'):
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=cfg['D'], nh=cfg['nh'], vocab=VOCAB,
                                 n_layer=cfg['n_layer'], steps=cfg['steps'],
                                 mlp_mult=cfg['mlp_mult'], W=cfg['W'])
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=cfg['D'], nh=cfg['nh'], vocab=VOCAB, n_layer=cfg['n_layer'])
    raise ValueError(arch)


def param_table(m):
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters()
              if n.split('.')[0] in ('e', 'embed', 'head', 'lm_head', 'h'))
    return tot, emb, tot - emb


def get_batch(data, B, T, device, gen=None):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix]).to(device)
    return x, y


@torch.no_grad()
def eval_split(m, data, device, B=32, T=1024, iters=40, seed=123):
    g = torch.Generator().manual_seed(seed)
    m.eval()
    tot, corr, n = 0.0, 0, 0
    for _ in range(iters):
        x, y = get_batch(data, B, T, device, g)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            lg, _ = m(x, None)
        lg = lg.float()
        tot += F.cross_entropy(lg.view(-1, VOCAB), y.view(-1), reduction='sum').item()
        corr += (lg.argmax(-1) == y).sum().item()
        n += y.numel()
    return tot / n, corr / n


@torch.no_grad()
def generate(m, tok, prompt, device, n_new=110, mode='greedy', temp=0.85, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    xg = torch.tensor([tok.encode(prompt).ids], device=device)
    m.eval()
    for _ in range(n_new):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            lg, _ = m(xg[:, -1024:], None)
        logits = lg[:, -1, :].float()
        if mode == 'greedy':
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
        xg = torch.cat([xg, nxt], 1)
    return tok.decode(xg[0].tolist())


def rep_ratio(s):
    """退化度：最长重复子串占比的粗略代理 = 最常见 8-gram 出现次数"""
    w = s.split()
    if len(w) < 16:
        return 0.0, ''
    from collections import Counter
    c = Counter(tuple(w[i:i + 6]) for i in range(len(w) - 5))
    g, k = c.most_common(1)[0]
    return k, ' '.join(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='path/ckpt.pt:arch')
    ap.add_argument('--b', required=True, help='path/ckpt.pt:arch')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = a.device

    tr = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
    va = np.memmap(VALID_BIN, dtype=np.uint16, mode='r')
    tok = Tokenizer.from_file(TOK_PATH)

    runs = []
    for spec in (a.a, a.b):
        path, arch = spec.rsplit(':', 1)
        ck = torch.load(path, map_location=dev, weights_only=False)
        cfg = ck.get('cfg') or ARCH_CFG[arch]
        m = build(arch, cfg).to(dev)
        m.load_state_dict(ck['model'])
        tot, emb, st = param_table(m)
        vl, vc = eval_split(m, va, dev)
        tl, tc = eval_split(m, tr, dev, iters=10)
        runs.append(dict(arch=arch, cfg=cfg, m=m, tot=tot, emb=emb, st=st,
                         vl=vl, vc=vc, tl=tl, tc=tc, path=path))
        # 释放干净
        del ck

    print('=' * 78)
    print(f"{'':<22}{'A: ' + runs[0]['arch']:>26}{'B: ' + runs[1]['arch']:>28}")
    print('-' * 78)
    rows = [('结构参数', 'st', '{:,}'.format), ('词表侧参数', 'emb', '{:,}'.format),
            ('总参数', 'tot', '{:,}'.format),
            ('val loss', 'vl', '{:.4f}'.format), ('val top-1', 'vc', '{:.2%}'.format),
            ('train loss', 'tl', '{:.4f}'.format), ('train top-1', 'tc', '{:.2%}'.format)]
    for label, k, f in rows:
        print(f"{label:<22}{f(runs[0][k]):>26}{f(runs[1][k]):>28}")
    print('=' * 78)

    for mode, kw in [('greedy', {}), ('sample(temp=0.85)', dict(mode='sample', temp=0.85))]:
        print(f"\n{'#' * 78}\n### 生成 [{mode}]\n{'#' * 78}")
        for pr in PROMPTS:
            print(f"\n>>> PROMPT: {pr}")
            for r in runs:
                out = generate(r['m'], tok, pr, dev, mode=kw.get('mode', 'greedy'),
                               temp=kw.get('temp', 1.0))
                body = out[len(pr):] if out.startswith(pr) else out
                k, gram = rep_ratio(body)
                flag = f'  [退化! 重复 6-gram x{k}: "{gram}"]' if k >= 4 else ''
                print(f"  --- {r['arch']} ---{flag}")
                print('   ' + body.replace('\n', ' ').strip()[:400])


if __name__ == '__main__':
    main()
