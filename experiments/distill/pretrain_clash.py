"""真实预训练对轰: la_cycle(steps=1) vs 同规模标准Transformer —— next-token 语言建模, 测真实能力。

为何: 爸爸指令"训真模型验证真能力"——KL 蒸馏分只是"模仿得分", 真实语言建模需
直接 next-token 预训练 + 验证集 perplexity(真实能力直接metric)。

军规:
- 学生 vs 学生? 这里是无教师直训(真预训练), 非蒸馏, 公平=同数据/同预算/同seed/同参数量级。
- 同规模: 用 param_budget 对齐三架构总参数(含词表), 使同量级。
- 先证学会(训练 loss 低于随机), 再用验证板 perplexity 比真实能力。
- 诚实: la_cycle 阶梯存在 O(T²) 注意(借 BDH), 复杂度另记账; 本节只比'真实语言能力'。

用法:  python pretrain_clash.py --arch la_cycle --data /data/dynfw/data/wikitext103.txt --out ... --temp 0.5
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

def build(arch, D, nh, n_layer, vocab, mlp_mult, block, steps=1, K=8):
    if arch == 'la_cycle':
        from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
        return BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=steps, mlp_mult=mlp_mult)
    if arch in ('v6', 'fusedfw_fw_cycle'):
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=steps,
                                 mlp_mult=mlp_mult, W=block)
    if arch == 'bdh':
        from dynfw.models.bdh_qwen import BDHQwen
        return BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab, dropout=0.0)
    if arch == 'bdh_rawfw':
        from dynfw.models.bdh_rawfw_qwen import BDHRawFWQwen
        return BDHRawFWQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab,
                            dropout=0.0, W=block)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=nh, n_layer=n_layer, vocab=vocab, maxT=block)
    raise ValueError(arch)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='la_cycle')
    ap.add_argument('--data', required=True)
    ap.add_argument('--teacher', default='/data/models/Qwen3-0.6B')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-batches', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--nh', type=int, default=16)
    ap.add_argument('--n-layer', type=int, default=2)
    ap.add_argument('--mlp-mult', type=int, default=64)
    ap.add_argument('--k', type=int, default=8, help='(unused; v7 DLA 已删除)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.01)
    ap.add_argument('--max-lines', type=int, default=10000, help='仅取前N行控制tokenize内存')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.pad_token = tok.eos_token
    # 关键: 模型词表必须用 config.vocab_size(=151936), 非 tokenizer.vocab_size(=151643)
    # 否则 Embedding 尺寸与真实权重词表不符(越界/错位)
    vocab = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype='auto').config.vocab_size
    print(f"model vocab_size={vocab}, tokenizer vocab_size={tok.vocab_size}")

    # 读数据 + 前几行 tokenize
    # 取数据前 N 行(控制内存; 用行数上限), 冒烟/对轰皆可切
    lines = [l.strip() for l in open(args.data) if l.strip()][:args.max_lines]
    # 打乱后切 train/val
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(lines))
    n_val = max(1, int(len(lines)*args.val_frac))
    val_lines = [lines[i] for i in idx[:n_val]]
    train_lines = [lines[i] for i in idx[n_val:]]
    print(f"total lines={len(lines)}, train={len(train_lines)}, val={len(val_lines)}")

    def make_batches(ls, max_b):
        # 批量 tokenize(快), 截断到 block, 组装 batch
        enc = tok(ls, return_tensors='pt', padding=True, truncation=True, max_length=args.block-1)
        ids = enc['input_ids']  # [N, L]
        # 过滤过短样本
        mask = (ids != tok.pad_token_id).sum(-1) >= 16
        ids = ids[mask]
        batches = ids.split(args.batch)
        return list(batches)[:max_b]

    train_batches = make_batches(train_lines, args.max_batches)
    val_batches = make_batches(val_lines, 20)
    print(f"train batches={len(train_batches)}, val batches={len(val_batches)}")

    m = build(args.arch, args.dim, args.nh, args.n_layer, vocab, args.mlp_mult, args.block, args.steps, K=args.k).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"=== arch={args.arch} params={m.np()/1e6:.1f}M ===")

    t0 = time.time()
    final_train_loss = None
    print(f"train batches={len(train_batches)}, val batches={len(val_batches)}")
    for ep in range(args.epochs):
        total, cnt = 0.0, 0
        for xb in train_batches:
            x = xb.cuda(); y = x[:, 1:].contiguous()
            lg, _ = m(x[:, :-1])
            loss = F.cross_entropy(lg.view(-1, vocab), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); cnt += 1
        print(f"  ep{ep} train_ppl={math.exp(total/cnt):.2f}")
        final_train_loss = total/cnt

    # 验证集 perplexity (真能力 metric)
    m.eval(); val_total, val_cnt = 0.0, 0
    with torch.no_grad():
        for xb in val_batches:
            x = xb.cuda(); y = x[:, 1:].contiguous()
            lg, _ = m(x[:, :-1])
            loss = F.cross_entropy(lg.view(-1, vocab), y.view(-1))
            val_total += loss.item(); val_cnt += 1
    val_ppl = math.exp(val_total/val_cnt)
    print(f"★★ {args.arch}: train_ppl={math.exp(final_train_loss):.2f}  val_ppl={val_ppl:.2f}  wall={time.time()-t0:.0f}s")

    res = {'arch': args.arch, 'params_M': round(m.np()/1e6,2), 'train_ppl': math.exp(final_train_loss),
           'val_ppl': val_ppl, 'epochs': args.epochs, 'seed': args.seed, 'wall_s': round(time.time()-t0,1)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(res, f, indent=2)
    # 保存 checkpoint + config (供 HF adapter/lm_eval 加载)
    torch.save(m.state_dict(), os.path.join(args.out, 'student.pt'))
    cfg = {'arch': args.arch, 'D': args.dim, 'nh': args.nh, 'n_layer': args.n_layer,
           'mlp_mult': args.mlp_mult, 'block': args.block, 'vocab': vocab,
           'steps': args.steps, 'k': args.k}
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print("[saved]", os.path.join(args.out, 'result.json'))
    print("[ckpt saved]", os.path.join(args.out, 'student.pt'), "config:", cfg)