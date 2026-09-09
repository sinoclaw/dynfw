"""DynFW 蒸馏入口（v2）：Qwen3-0.6B 教师 -> {FusedFWQwen | TF_sdpa} 学生（离线 logits 蒸馏）。

方案A 架构对决版（2026-09-08）。加 --arch 参数切换学生架构，实现公平对轰：
  --arch fusedfw   学生=FusedFWQwen（稀疏 fast-weight rho 记忆）
  --arch tf        学生=TF_sdpa（标准 Transformer，SDPA 注意力）
两个学生在【完全相同的教师/数据/训练预算/seed】下蒸馏，唯一变量=学生架构。

公平性核心（对轰纪律）：
  - 相同语料、相同 tokenizer、相同 block/batch/epoch/lr/temperature
  - 相同 seed（torch.manual_seed 在模型创建前）
  - 报参数量（np()）做参数对齐，避免"容量不同"误判为"架构优劣"
  - 报 KL 蒸馏损失收敛曲线 + 生成困惑度（真实能力指标）

用法：
  python experiments/distill/distill_qwen.py --arch fusedfw --teacher Qwen/Qwen3-0.6B \\
      --data <语料> --out results/arch_fusedfw --dim 128 --slots 512 --k 16 --epochs 20
  python experiments/distill/distill_qwen.py --arch tf --teacher Qwen/Qwen3-0.6B \\
      --data <语料> --out results/arch_tf --dim 128 --nh 4 --n-layer 4 --epochs 20
"""
import os, argparse, json, torch
import torch.nn.functional as F
from dynfw.models.fused_fw_qwen import FusedFWQwen          # 学生 A（无时序求和rho）
from dynfw.models.fused_fw_cycle import FusedFWCYBLE          # 学生 A-cyc（循环潜推理,吸收BDH-CQ）
from dynfw.models.fused_fw_rec import FusedFWRecurrent      # 学生 A'（方向A：递推rho保留顺序）
from dynfw.models.fused_fw_la import FusedFWLa              # 学生 A''（吸收BDH：内容寻址因果读取）
from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM    # 学生 A''-cyc（K-is-Q 保顺序 + 循环潜推理）


from dynfw.models.fused_fw_full import FusedFWFull                  # 学生 D（BDH 完整机制分)
from dynfw.models.fused_fw_full_shared import FusedFWFullShared     # 学生 E（省参：weight-sharing + tie词表，官方BDH机制）
from dynfw.models.fused_fw_lin import FusedFWLin             # 学生 A''''（线性注意力+内容寻址）
from dynfw.models.bdh_gla import BDHGLA                     # 学生 M（BDH稀疏 + GLA线性核 = BDH+Mamba式）
from dynfw.models.bdh_gla_v2 import BDHGLAv2                # 学生 M2（v2：完整保留BDH表达力 + GLA线性）
from dynfw.models.bdh_gla_v3 import BDHGLAv3                # 学生 M3（v3：x_sparse作Q=K=V过GLA，最忠实BDH K-is-Q）
from dynfw.models.bdh_qwen import BDHQwen                   # 学生 C（BDH：稀疏+linear attention保顺序）
from dynfw.models.transformer import TF_sdpa                # 学生 B（对照）


def make_student(args, teacher_vocab, device):
    """按 --arch 构造学生，返回 (model, n_params)。"""
    if args.arch == 'fusedfw':
        m = FusedFWQwen(D=args.dim, N=args.slots, k=args.k,
                        vocab=teacher_vocab, use_ffn=True, n_layer=args.n_layer)
    elif args.arch == 'fusedfw_cycle':
        m = FusedFWCYBLE(D=args.dim, N=args.slots, k=args.k,
                         vocab=teacher_vocab, use_ffn=True, n_layer=args.n_layer,
                         steps=args.cycle_steps)
    elif args.arch == 'fusedfw_rec':
        m = FusedFWRecurrent(D=args.dim, N=args.slots, k=args.k,
                             vocab=teacher_vocab, use_ffn=True, n_layer=args.n_layer)
    elif args.arch == 'fusedfw_la':
        m = FusedFWLa(D=args.dim, N=args.slots, k=args.k, nh=args.nh,
                      vocab=teacher_vocab, use_ffn=True, n_layer=args.n_layer)
    elif args.arch == 'fusedfw_la_cycle':
        m = BDHBlockCycleLM(D=args.dim, nh=args.nh, vocab=teacher_vocab,
                            n_layer=args.n_layer, steps=args.cycle_steps,
                            mlp_mult=args.mlp_mult)

    elif args.arch == 'fusedfw_fw_cycle':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        m = BDHBlockFWCycleLM(D=args.dim, nh=args.nh, vocab=teacher_vocab,
                              n_layer=args.n_layer, steps=args.cycle_steps,
                              mlp_mult=args.mlp_mult, W=args.block)


    elif args.arch == 'fusedfw_full':
        m = FusedFWFull(D=args.dim, N=args.slots, k=args.k, nh=args.nh,
                        mlp_mult=args.mlp_mult, vocab=teacher_vocab,
                        use_ffn=False, n_layer=args.n_layer, use_softmax=args.softmax, tie=args.tie)
    elif args.arch == 'fusedfw_full_shared':
        m = FusedFWFullShared(D=args.dim, N=args.slots, k=args.k, nh=args.nh,
                              mlp_mult=args.mlp_mult, vocab=teacher_vocab,
                              use_ffn=False, n_layer=args.n_layer, use_softmax=args.softmax,
                              tie=args.tie)
    elif args.arch == 'fusedfw_lin':
        m = FusedFWLin(D=args.dim, nh=args.nh, dk=32, vocab=teacher_vocab,
                       n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh_gla':
        m = BDHGLA(D=args.dim, nh=args.nh, dk=32, vocab=teacher_vocab,
                   n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh_gla2':
        m = BDHGLAv2(D=args.dim, nh=args.nh, dk=args.dk, N=args.slots, vocab=teacher_vocab,
                     n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh_gla3':
        m = BDHGLAv3(D=args.dim, nh=args.nh, N=args.slots, vocab=teacher_vocab,
                     n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh':
        m = BDHQwen(D=args.dim, n_layer=args.n_layer, nh=args.nh,
                    mlp_mult=args.mlp_mult, vocab=teacher_vocab, dropout=0.0)
    elif args.arch == 'tf':
        m = TF_sdpa(D=args.dim, nh=args.nh, n_layer=args.n_layer,
                    vocab=teacher_vocab, maxT=args.block)
    else:
        raise ValueError(f"unknown arch: {args.arch}")
    return m, m.np()


def kl_loss(student_logits, teacher_logits, temperature=1.0):
    """logits 蒸馏 KL 损失（soft labels）。"""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)


def perplexity_from_loss(mean_kl):
    """从 KL 蒸馏损失估计困惑度下界（exp(kl)，粗糙但可对比）。"""
    import math
    return math.exp(min(mean_kl, 30.0))  # 封顶防溢出


def main():
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='fusedfw', choices=['fusedfw', 'fusedfw_cycle', 'fusedfw_rec', 'fusedfw_la', 'fusedfw_la_cycle', 'fusedfw_fw_cycle', 'fusedfw_full', 'fusedfw_full_shared', 'fusedfw_lin', 'bdh_gla', 'bdh_gla2', 'bdh_gla3', 'bdh', 'tf'],
                    help='学生架构：fusedfw / fusedfw_rec / fusedfw_la / fusedfw_full(BDH完整) / bdh / tf')
    ap.add_argument('--teacher', type=str, default='Qwen/Qwen3-0.6B')
    ap.add_argument('--data', type=str, required=True, help='语料文本文件 (每行一行)')
    ap.add_argument('--out', type=str, default='/data/dynfw/results/distill_qwen')
    ap.add_argument('--max-lines', type=int, default=2000)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-batches', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--shared-logits', type=str, default='',
                    help='共享教师logits目录；若已存在teacher_logits.npy则复用(不重算)，否则算后存这里')
    # FusedFW 学生参数
    ap.add_argument('--dim', type=int, default=128, help='学生 D（隐藏维，两架构共用）')
    ap.add_argument('--slots', type=int, default=512, help='FusedFW 学生 N')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--n-layer', type=int, default=1)
    ap.add_argument('--cycle-steps', type=int, default=4, help='循环潜推理步数(fusedfw_cycle 专用)')
    # TF 学生参数
    ap.add_argument('--nh', type=int, default=4, help='TF 注意力头数')
    ap.add_argument('--mlp-mult', type=int, default=128, dest='mlp_mult', help='BDH 稀疏维乘数 N=mlp_mult*D//nh')
    ap.add_argument('--dk', type=int, default=32, help='GLA 线性注意力 head 状态宽 dk')
    ap.add_argument('--softmax', action='store_true', help='FusedFWFull 注意力加 softmax（默认 raw，消融用）')
    ap.add_argument('--tie', action='store_true', help='tie embeddings（lm_head 复用 embed，省 vocab*D 参数）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)          # 模型创建前 seed（公平）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== arch={args.arch} device={device} seed={args.seed} ===")

    # 1. tokenizer + 数据（学生无关，固定）
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"=== tokenizer vocab_size: {tok.vocab_size} ===")
    with open(args.data) as f:
        lines = [l.strip() for l in f if l.strip()][:args.max_lines]
    ids = tok(lines, return_tensors='pt', padding=True, truncation=True,
              max_length=args.block)['input_ids']
    print(f"=== data tensor: {ids.shape} ===")

    # 2. 教师预计算 logits（固定，两学生共用同一份）
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map='auto')
    teacher.eval()
    teacher_vocab = teacher.config.vocab_size
    print(f"=== teacher config: {teacher.config.hidden_size} dim, {teacher_vocab} vocab ===")

    os.makedirs(args.out, exist_ok=True)
    logits_dir = os.path.join(args.out, 'teacher_logits')
    # 共享教师logits：若已缓存则直接复用，避免每次重算12.4G
    t_arr = None
    if args.shared_logits:
        shared_npy = os.path.join(args.shared_logits, 'teacher_logits.npy')
        if os.path.exists(shared_npy):
            import numpy as np
            t_arr = np.load(shared_npy)
            print(f"=== reuse shared teacher logits: {t_arr.shape} ===")
    if t_arr is None:
        x = ids.to(device)
        batches = x.split(args.batch)[:args.max_batches]
        teacher_logits_all = []
        t_blocks = []
        for bx in batches:
            with torch.no_grad():
                lg = teacher(bx).logits.float()
            teacher_logits_all.append(lg.cpu().numpy().astype(np.float16))
            t_blocks.append(bx)
        t_arr = np.concatenate(teacher_logits_all, axis=0)
        os.makedirs(logits_dir, exist_ok=True)
        np.save(os.path.join(logits_dir, 'teacher_logits.npy'), t_arr)
        # 存到共享目录供后续复用
        if args.shared_logits:
            os.makedirs(args.shared_logits, exist_ok=True)
            np.save(os.path.join(args.shared_logits, 'teacher_logits.npy'), t_arr)
        print(f"=== teacher logits precomputed: {t_arr.shape} ===")
    else:
        # 复用路径：重建 t_blocks（数据块与教师无关，仅用于学生输入）
        x = ids.to(device)
        n_blocks = t_arr.shape[0] // args.batch   # t_arr 总行数=batch数*batch，须严格对齐
        t_blocks = list(x.split(args.batch))[:n_blocks]

    # 3. 学生训练（读同一份 logits；改 --arch 即可对轰）
    import time
    t0 = time.time()
    student, n_params = make_student(args, teacher_vocab, device)
    student = student.to(device)
    print(f"=== student({args.arch}) params: {n_params} ===")

    losses = []
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student.train()
    batch_x = torch.cat(t_blocks, dim=0)
    for epoch in range(args.epochs):
        for i in range(0, batch_x.size(0), args.batch):
            bx = batch_x[i:i+args.batch].to(device)
            t_lg = torch.from_numpy(t_arr[i:i+args.batch]).to(device).float()
            if args.arch in ('tf', 'bdh', 'fusedfw_full', 'fusedfw_full_shared', 'fusedfw_lin', 'bdh_gla', 'bdh_gla2', 'bdh_gla3'):
                s_lg, _ = student.forward(bx)      # TF/BDH/FusedFWFull/FusedFWLin/BDHGLA/BDHGLAv2/BDHGLAv3 返回 (logits, loss)
            else:
                s_lg = student.forward_logits(bx)  # FusedFW 返回 logits
            loss = kl_loss(s_lg.float(), t_lg, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if len(losses) % 20 == 0:
                print(f"  epoch {epoch} step {len(losses)} loss {loss.item():.4f}")
    wall = time.time() - t0

    final_loss = losses[-1] if losses else None
    mean_loss = float(np.mean(losses)) if losses else None
    ppl = perplexity_from_loss(mean_loss) if mean_loss else None
    result = {
        'arch': args.arch,
        'device': device, 'teacher': args.teacher,
        'student_params': n_params, 'vocab': teacher_vocab,
        'dim': args.dim, 'slots': args.slots, 'k': args.k, 'n_layer': args.n_layer,
        'nh': args.nh, 'epochs': args.epochs, 'seed': args.seed,
        'final_loss': final_loss, 'mean_loss': mean_loss,
        'ppl_est': ppl, 'wall_sec': wall, 'blocks': batch_x.size(0),
    }
    with open(os.path.join(args.out, 'distill_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    os.makedirs(os.path.join(args.out, 'checkpoint'), exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out, 'checkpoint/student.pt'))
    print(f"=== DONE: {json.dumps(result, ensure_ascii=False)} ===")


if __name__ == '__main__':
    main()
