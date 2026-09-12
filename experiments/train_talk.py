"""小模型「会说话」训练：我们架构(v6/v6w/cfg) vs 标准 SDPA-Transformer，
同结构参数+同深度+同预算+同seed。

用法:
  python train_talk.py --arch v6|tf|v6w --steps N --out results/talk_<arch>
  # N/D 消融：固定结构参 = 3*mm*D^2*L，只动 nh 改 N = mm*D/nh
  python train_talk.py --arch cfg --D 256 --nh 4 --mm 4 --steps 8000 --batch 32 --T 1024 \
                       --out results/nd_nh4

⚠️ eval 口径：原版 eval_loss 的 get_batch 不固定 seed，导致同一 ckpt 的 val_loss
   每次不同（实测同一 TF ckpt 1.5002 vs 1.4882，差 0.012 = 抽样噪声）。
   新增 --eval-seed（默认 None = 保持旧行为）；做配置间比较时必须固定它。
"""
import argparse, math, os, sys, time, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.transformer import TF_sdpa

DATA = '/data/corpus/tinystories'
TOK  = '/data/models/gpt2-tok/tokenizer.json'

def build(arch, vocab=50257, D=None, nh=None, mm=None):
    if arch == 'v6':
        # D=128 nh=8 mm=16 L=6 : 每层 3*mm*D^2 = 3*16*16384 = 786K -> 6层 4.72M
        return BDHBlockFWCycleLM(D=128, nh=8, vocab=vocab, n_layer=6, steps=1, mlp_mult=16, W=256)
    elif arch == 'v6w':
        # 同总参臂：D=256 mm=4 -> 每层 3*4*256^2 = 786K -> 6层 4.72M 结构参
        #            词表侧 50257*256*2 = 25.73M -> 总参 30.45M（对齐 TF 的 30.52M）
        return BDHBlockFWCycleLM(D=256, nh=8, vocab=vocab, n_layer=6, steps=1, mlp_mult=4, W=256)
    elif arch == 'tf':
        # D=256 nh=8 L=6 : 每层 (3+1+4+4)*D^2 = 12*65536 = 786K -> 6层 4.72M
        return TF_sdpa(D=256, nh=8, n_layer=6, vocab=vocab)
    elif arch == 'cfg':
        # N/D 消融臂：结构参 3*mm*D^2*6 恒定，N = mm*D/nh 随 nh 变
        return BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=6, steps=1, mlp_mult=mm, W=256)
    raise ValueError(arch)

def param_table(m, vocab):
    tot = sum(p.numel() for p in m.parameters())
    emb = 0
    for n, p in m.named_parameters():
        if n in ('e.weight', 'h.weight', 'embed.weight', 'lm_head') or 'embed' in n or 'head' in n:
            emb += p.numel()
    return dict(total=tot, vocab_side=emb, struct=tot - emb)

def get_batch(data, B, T, device, gen=None):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+T].astype(np.int64)) for i in ix]).to(device)
    return x, y

@torch.no_grad()
def eval_loss(model, data, B, T, device, iters=40, seed=None):
    """seed=None → 旧行为（不固定，有抽样噪声）；seed=int → 固定口径，可跨配置比较"""
    model.eval(); ls = []
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    for _ in range(iters):
        x, y = get_batch(data, B, T, device, gen)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        ls.append(loss.item())
    model.train(); return sum(ls) / len(ls)

@torch.no_grad()
def sample(model, tok, prompts, n=120, T=1024, device='cuda', max_new=110):
    model.eval(); out = []
    for pr in prompts:
        ids = tok.encode(pr).ids
        x = torch.tensor([ids], device=device)
        for _ in range(max_new):
            xc = x[:, -T:]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, _ = model(xc, None)
            nxt = lg[:, -1, :].float().argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
        out.append(tok.decode(x[0].tolist()))
    model.train(); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['v6', 'tf', 'v6w', 'cfg'])
    ap.add_argument('--D', type=int, default=256)
    ap.add_argument('--nh', type=int, default=8)
    ap.add_argument('--mm', type=int, default=4)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-seed', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log-every', type=int, default=250)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    os.makedirs(a.out, exist_ok=True)
    log = open(f'{a.out}/train.log', 'w')

    def P(*xs):
        s = ' '.join(str(x) for x in xs); print(s, flush=True); log.write(s + '\n'); log.flush()

    tr = np.memmap(f'{DATA}/train.bin', dtype=np.uint16, mode='r')
    va = np.memmap(f'{DATA}/valid.bin', dtype=np.uint16, mode='r')
    tok_tokens = a.steps * a.batch * a.T
    hdr = f'=== arch={a.arch} seed={a.seed} steps={a.steps} batch={a.batch} T={a.T}'
    if a.arch == 'cfg':
        hdr += f' D={a.D} nh={a.nh} mm={a.mm} N={a.mm*a.D//a.nh} N/D={a.mm/a.nh:.3f}'
    P(hdr + ' ===')
    P(f'[data] train={len(tr):,} val={len(va):,} tokens | 本次消耗={tok_tokens:,} ({tok_tokens/len(tr):.2f} epoch)')
    P(f'[eval] eval_seed={a.eval_seed} (None=不固定, 有抽样噪声)')

    model = build(a.arch, D=a.D, nh=a.nh, mm=a.mm).to(dev)
    pt = param_table(model, 50257)
    P(f"[params] total={pt['total']/1e6:.3f}M  vocab_side={pt['vocab_side']/1e6:.3f}M  struct={pt['struct']/1e6:.3f}M")
    P(f"[params] n_layer={getattr(model,'n_layer',None)}  struct_exact={pt['struct']}")
    if tok_tokens / len(tr) > 4: P('!!! 超过 4 epoch 硬规则'); sys.exit(1)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, 0.95))
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        prog = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
    for g in opt.param_groups: g['initial_lr'] = a.lr

    t0 = time.time(); hist = []
    for s in range(a.steps):
        for g in opt.param_groups: g['lr'] = lr_at(s)
        x, y = get_batch(tr, a.batch, a.T, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % a.log_every == 0 or s == a.steps - 1:
            el = time.time() - t0
            P(f'step {s:>6}/{a.steps}  loss {loss.item():.4f}  lr {lr_at(s):.2e}  gnorm {gn:.2f}  {el:.0f}s  {tok_tokens*(s+1)/a.steps/el/1e3:.1f}k tok/s')
    train_t = time.time() - t0
    vl = eval_loss(model, va, a.batch, a.T, dev, seed=a.eval_seed)
    P(f'[DONE] wall={train_t:.0f}s  val_loss={vl:.4f}  val_ppl={math.exp(vl):.1f}')
    torch.save({'model': model.state_dict(), 'arch': a.arch, 'val_loss': vl,
                'params': pt, 'steps': a.steps, 'seed': a.seed,
                'cfg': dict(D=a.D, nh=a.nh, mm=a.mm, N=a.mm*a.D//a.nh)}, f'{a.out}/ckpt.pt')

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK)
    prompts = ['Once upon a time, there was a little girl named Lily.',
               'One day, a small dog went to the park.',
               'Tom was hungry, so he']
    gens = sample(model, tk, prompts, device=dev)
    with open(f'{a.out}/samples.txt', 'w') as fh:
        for pr, g in zip(prompts, gens):
            fh.write(f'--- PROMPT: {pr}\n{g}\n\n')
    for pr, g in zip(prompts, gens):
        P(f'--- PROMPT: {pr}'); P(g); P('')
    json.dump({'arch': a.arch, 'val_loss': vl, 'val_ppl': math.exp(vl), 'params': pt,
               'wall_s': train_t, 'steps': a.steps, 'tokens': tok_tokens,
               'eval_seed': a.eval_seed, 'cfg': dict(D=a.D, nh=a.nh, mm=a.mm, N=a.mm*a.D//a.nh)},
              open(f'{a.out}/summary.json', 'w'), indent=1)
    log.close()

if __name__ == '__main__':
    main()
