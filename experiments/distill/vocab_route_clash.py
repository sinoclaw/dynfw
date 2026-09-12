"""词表路线对比实验：同架构 v6/v7，不同词表预训练，比语言建模损失(BPB)。
目的：回答『千问词表 vs 自训 vs byte-level 对我们架构的影响』。
军规：同架构/同语料/同预算/同 seed，按每字节损失(BPB)归一对比(因字节序列更长)。
"""
import argparse, math, os, time
import torch

def make_model(arch, vocab, D=128, nh=16, n_layer=2, mlp_mult=64, K=8):
    if arch == 'v6':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256)
    elif arch == 'v7':
        from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
        return BDHBlockDLACycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=1, mlp_mult=mlp_mult, W=256, K=K)
    else:
        raise ValueError(arch)

def tokenize_bytes(text_file, max_chars=2_000_000, block=256):
    """byte-level tokenize: text -> utf-8 bytes, chunk to block"""
    raw = open(text_file, 'r', encoding='utf-8', errors='ignore').read(max_chars)
    data = raw.encode('utf-8')
    seqs = [list(data[i:i+block]) for i in range(0, len(data)-block, block)]
    return torch.tensor(seqs)  # [N, block] of byte values

def train_lm(model, data, vocab, arch, epochs=3, lr=3e-4, batch=8, seed=0, use_emb=False):
    """自训练语言建模(非蒸馏): 预测下一个 token, 返回平均 loss."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nll_total = n_tokens = 0
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0])
        for i in range(0, data.shape[0], batch):
            idx = perm[i:i+batch]
            if idx.shape[0] < 2:
                continue
            x = data[idx].long()
            if use_emb:
                # 用 HF 词表 embedding——这里简化: 直接 cast 到 vocab 范围(A/B 组用)
                pass
            x = torch.clamp(x, 0, vocab-1)
            inpt, tgt = x[:, :-1], x[:, 1:]
            inpt = inpt.contiguous(); tgt = tgt.contiguous()
            model.train()
            loss = model(inpt, targets=tgt)[1]  # forward(x, targets) 返回 (lg, loss)
            if loss is None:
                # fallback: 手动算 cross_entropy
                lg = model(inpt, targets=None)[0]
                loss = torch.nn.functional.cross_entropy(lg.view(-1, model.vocab), tgt.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nll_total += loss.item() * tgt.numel()
            n_tokens += tgt.numel()
    return nll_total / max(n_tokens, 1)  # per-token NLL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='v6', choices=['v6', 'v7'])
    ap.add_argument('--data', default='/data/dynfw/data/wikitext103.txt')
    ap.add_argument('--vocab', type=int, default=256, help='256=byte-level; 151936=qwen; 65536/131072=自训')
    ap.add_argument('--tokenizer', type=str, default='', help='若有则用 HF tokenizer 分词(A/B组), 空则byte')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.tokenizer:
        # A/B 组: 用 HF 词表分词 (来自千问或自训)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        lines = [l.strip() for l in open(args.data, encoding='utf-8', errors='ignore') if l.strip()][:5000]
        enc = tok(lines, return_tensors='pt', padding=True, truncation=True, max_length=args.block)
        data = enc['input_ids']
        tok_pad = tok.pad_token_id if tok.pad_token is not None else 0
        vocab = tok.vocab_size if hasattr(tok, 'vocab_size') else tok.pad_token_id
        print(f"[A/B组] tokenizer={args.tokenizer} vocab={tok.vocab_size} data={data.shape}")
        use_emb = True
        # 注意: 模型词表必须 = tokenizer vocab (A/B组) 或 =args.vocab (byte)
        vocab = max(tok.vocab_size, 151936 if 'Qwen' in args.tokenizer else tok.vocab_size)
    else:
        # C 组: byte-level
        data = tokenize_bytes(args.data, block=args.block)
        vocab = args.vocab
        print(f"[C组 byte] vocab={vocab} data={data.shape}")
        use_emb = False

    model = make_model(args.arch, vocab).to(device)
    print(f"=== arch={args.arch} vocab={vocab} params={round(model.np()/1e6,1)}M seed={args.seed} ===")

    # 训练
    avg_nll = train_lm(model, data.to(device), vocab, args.arch,
                       epochs=args.epochs, batch=args.batch, seed=args.seed, use_emb=use_emb)
    # BPB 归一: BPB = NLL / (每token平均字节数). byte组每token=1字节 -> BPB=NLL.
    # A/B组: 用原始语料测 平均bytes/token
    if args.tokenizer:
        import numpy as np
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        lines = [l.strip() for l in open(args.data, encoding='utf-8', errors='ignore') if l.strip()][:2000]
        tot_byte = sum(len(l.encode('utf-8')) for l in lines)
        tot_tok = sum(len(tok(l, add_special_tokens=False).input_ids) for l in lines)
        bytes_per_tok = tot_byte / max(tot_tok, 1)
    else:
        bytes_per_tok = 1.0
    bpb = avg_nll / bytes_per_tok
    ppl = math.exp(min(avg_nll, 20))
    print(f"final NLL/token={avg_nll:.4f}  bytes_per_tok={bytes_per_tok:.2f}  BPB={bpb:.4f}  PPL={ppl:.2f}")
    print(f"RESULT arch={args.arch} vocab={vocab} params={round(model.np()/1e6,1)}M nll={avg_nll:.4f} bpb={bpb:.4f} ppl={ppl:.2f}")

if __name__ == '__main__':
    main()
