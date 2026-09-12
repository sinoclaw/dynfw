"""小规模三方对轰: la_cycle vs bdh_qwen vs TF_sdpa (同结构参数口径)。

目的(爸爸指令): 先在 4090 可快速跑的小规模钉死 la_cycle 相对同规模标准 Llama/BDH
是否真有架构优势(高智能密度); 有优势才上卡放大训'小钢炮'。

军规:
- 同结构参数对轰; 同 teacher=Qwen3-0.6B / 同数据 / 同预算 / 同 seed; 学生 vs 学生。
- 判据先用训练 loss 证学会(低于随机), 再比 KL 蒸馏 loss(越低越好)。
- 如实标注: 三者都 O(T²)(la_cycle/bdh 借 BDH 二维注意, TF 标准注意), 故这是'同量级参数
  谁更强'的能力对轰, 不是'自家 O(T) 差异化'之争——O(T) 优势另记。

参数对齐注意:
- TF_sdpa 每层结构参数 ≈ 12·D² (qkv 3D² + proj D² + FFN 4D² + down 4D²)
- BDH/la_cycle 每层 ≈ encoder(hoD·N) + encoder_v(hoD·N) + decoder(hoN·D), N=mlp_mult·D//hn
  la_cycle 多循环(同权重复用, 结构参数不变, 只多计算)。
- 用同 D, 调 n_layer 让结构参数尽量接近同一预算。
"""
import json, math, time
import torch
import torch.nn.functional as F


def build(arch, D, nh, n_layer, vocab, mlp_mult, block, steps=2):
    if arch == 'la_cycle':
        from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
        return BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=steps, mlp_mult=mlp_mult)
    if arch == 'bdh':
        from dynfw.models.bdh_qwen import BDHQwen
        return BDHQwen(D=D, n_layer=n_layer, nh=nh, mlp_mult=mlp_mult, vocab=vocab, dropout=0.0)
    if arch == 'tf':
        from dynfw.models.transformer import TF_sdpa
        return TF_sdpa(D=D, nh=nh, n_layer=n_layer, vocab=vocab, maxT=block)
    raise ValueError(arch)


def n_params(m):
    return m.np()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=128)
    ap.add_argument('--nh', type=int, default=16)
    ap.add_argument('--teacher', default='/data/models/Qwen3-0.6B')
    ap.add_argument('--data', required=True)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-batches', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--mlp-mult', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--teacher-logits', default='/data/dynfw/results/clash_fusedfw/teacher_logits')
    args = ap.parse_args()
    torch.manual_seed(args.seed); import numpy as np; np.random.seed(args.seed)

    vocab = 151936
    # 加载数据
    raw = open(args.data).read().split('\n')
    raw = [x for x in raw if x.strip()][:args.max_batches*args.batch]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.pad_token = tok.eos_token
    enc = []
    for line in raw:
        ids = tok.encode(line)[:args.block]
        if len(ids) < 8:
            continue
        enc.append(ids)
    # 不够就 padding 拼
    batches = []
    for i in range(0, len(enc)-args.batch+1, args.batch):
        b = enc[i:i+args.batch]
        maxl = min(max(len(x) for x in b), args.block)
        b = [x[:maxl] + [tok.eos_token]*(maxl-len(x[:maxl])) for x in b]
        batches.append(torch.tensor(b, dtype=torch.long))
    batches = batches[:args.max_batches]

    # 教师 logits 复用 (npy 缓存)
    tnpy = os.path.join(args.teacher_logits, 'teacher_logits.npy')
    import numpy as np
    tl = np.load(tnpy)  # shape? 见 cash
    print("teacher logits shape:", tl.shape)
    teacher_logits = torch.from_numpy(tl).float()  # 需要与 batches 对齐

    # 三种架构对齐 (同 D, 调 n_layer 让结构参数接近)
    arch_cfg = {}
    for arch, nl in [('la_cycle', 2), ('bdh', 2), ('tf', 2)]:
        m = build(arch, args.D, args.nh, nl, vocab, args.mlp_mult, args.block)
        arch_cfg[arch] = (nl, n_params(m))
        print(f"  {arch}: D={args.D} L={nl} 结构参数={n_params(m)/1e6:.1f}M")

    # 训练对比
    out = {}
    for arch, (nl, np_) in arch_cfg.items():
        m = build(arch, args.D, args.nh, nl, vocab, args.mlp_mult, args.block).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        print(f"\n=== 训练 {arch} ===")
        t0 = time.time()
        final_kl = None
        for ep in range(args.epochs):
            for bi, xb in enumerate(batches):
                x = xb.cuda(); y = x[:, 1:]
                lg, _ = m(x[:, :-1])
                # 教师 logits 截取对应位置 (需与 y 对齐)
                tl_b = teacher_logits[bi].cuda() if bi < teacher_logits.shape[0] else None
                if tl_b is None:
                    continue
                tl_slice = tl_b[:x.shape[1]-1]  # 简化: 假设 tl_b 是 [T,vocab]
                kl = kl_div(lg, tl_slice)
                opt.zero_grad(); kl.backward(); opt.step()
                final_kl = kl.item()
        out[arch] = {'n_params_M': round(np_/1e6, 2), 'final_kl': final_kl,
                     'wall_s': round(time.time()-t0, 1)}
        print(f"  ★ {arch}: 结构参数={np_/1e6:.1f}M  final_kl={final_kl:.2f}  wall={time.time()-t0:.0f}s")
        del m; torch.cuda.empty_cache()

    print("\n=== 对轰结论 ===")
    for k, v in out.items():
        print(f"  {k}: {v['n_params_M']}M, final_kl={v['final_kl']}")
    with open('/tmp/clash_small.json', 'w') as f:
        json.dump(out, f, indent=2)


def kl_div(student_logits, teacher_logits, temp=1.0):
    s = F.log_softmax(student_logits / temp, dim=-1)
    t = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(s, t, reduction='batchmean')
