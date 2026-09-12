"""千问词表 vs byte-level 两方公平对轰（同结构参数预算, 同语料, 同seed）。
军规: 词表对比 同结构参数(同D/同L/同mlp) → 只有词表税不同，比语言建模 BPB。
关键: byte 组词表税≈0, 同样结构参数下总参更小——回答"词表税参数值不值"。
"""
import argparse, math, os, torch

def make_arch_model(arch, vocab, D, n_layer, nh, mlp_mult):
    if arch == 'v6':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256)
    elif arch == 'v7':
        from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
        return BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256, K=8)

def tokenize_bytes(text_file, max_chars=3_000_000, block=256):
    raw = open(text_file, 'r', encoding='utf-8', errors='ignore').read(max_chars)
    data = raw.encode('utf-8')
    seqs = [list(data[i:i+block]) for i in range(0, len(data)-block, block)]
    return torch.tensor(seqs).long()

def train_lm(model, data, vocab, epochs=3, lr=3e-4, batch=8, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nll_total = n_tokens = 0
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0])
        for i in range(0, data.shape[0], batch):
            idx = perm[i:i+batch]
            if idx.shape[0] < 2: continue
            x = torch.clamp(data[idx], 0, vocab-1)
            inpt, tgt = x[:, :-1].contiguous(), x[:, 1:].contiguous()
            model.train()
            loss = model(inpt, targets=tgt)[1]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nll_total += loss.item() * tgt.numel()
            n_tokens += tgt.numel()
    return nll_total / max(n_tokens, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='v6', choices=['v6', 'v7'])
    ap.add_argument('--data', default='/data/dynfw/data/wikitext103.txt')
    ap.add_argument('--D', type=int, default=128)
    ap.add_argument('--n-layer', type=int, default=2)
    ap.add_argument('--mlp-mult', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 字节数据 (两方共用: byte组直接喂; 千问组用它作BPB参考)
    data_byte = tokenize_bytes(args.data, block=256).to(device)

    results = {}
    # 1) byte-level 组
    for tag, vocab in [("byte", 256), ("qwen151k", 151936)]:
        m = make_arch_model(args.arch, vocab, args.D, args.n_layer, 16, args.mlp_mult).to(device)
        struct = m.np() - 2 * vocab * args.D
        data = data_byte if vocab == 256 else data_byte  # 字节序列喂两边 (千问组clamp到151K内, byte值<256安全)
        avg_nll = train_lm(m, data, vocab, epochs=args.epochs, batch=args.batch, seed=args.seed)
        results[tag] = (m.np(), struct, avg_nll)
        print(f"[{tag}] {args.arch} vocab={vocab} D={args.D} L={args.n_layer} mlp={args.mlp_mult} "
              f"params={m.np()/1e6:.1f}M  struct={struct/1e6:.2f}M  NLL/tok={avg_nll:.4f}")
        del m
        torch.cuda.empty_cache()

    # 对比 (同序列, 同结构参数)
    (p_a, s_a, n_a) = results["byte"]
    (p_b, s_b, n_b) = results["qwen151k"]
    print(f"\n=== 同结构参数: byte({s_a/1e6:.1f}M) vs qwen({s_b/1e6:.1f}M) ===")
    print(f"  byte: 总参={p_a/1e6:.1f}M NLL={n_a:.4f}")
    print(f"  qwen: 总参={p_b/1e6:.1f}M NLL={n_b:.4f}  (词表税={2*151936*args.D/1e6:.1f}M)")
    print(f"  词表税代价: qwen多花 {2*151936*args.D/1e6:.1f}M 参数, NLL{'更高(差)' if n_b>n_a else '更低(好)'} {abs(n_b-n_a):.4f}")

if __name__ == '__main__':
    main()
