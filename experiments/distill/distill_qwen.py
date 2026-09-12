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


from dynfw.models.fused_fw_full import FusedFWFull                  # 学生 D（BDH 完整机制分)
from dynfw.models.fused_fw_full_shared import FusedFWFullShared     # 学生 E（省参：weight-sharing + tie词表，官方BDH机制）
from dynfw.models.fused_fw_lin import FusedFWLin             # 学生 A''''（线性注意力+内容寻址）
try:                                                         # GLA 线依赖 fla；缺 fla 时其余架构照常可用
    from dynfw.models.bdh_gla import BDHGLA                 # 学生 M（BDH稀疏 + GLA线性核 = BDH+Mamba式）
    from dynfw.models.bdh_gla_v2 import BDHGLAv2            # 学生 M2（v2：完整保留BDH表达力 + GLA线性）
    from dynfw.models.bdh_gla_v3 import BDHGLAv3            # 学生 M3（v3：x_sparse作Q=K=V过GLA，最忠实BDH K-is-Q）
    _GLA_ERR = None
except ImportError as _e:                                    # ModuleNotFoundError: fla
    BDHGLA = BDHGLAv2 = BDHGLAv3 = None
    _GLA_ERR = _e
from dynfw.models.bdh_qwen import BDHQwen                   # 学生 C（BDH：稀疏+linear attention保顺序）
from dynfw.models.transformer import TF_sdpa                # 学生 B（对照）
from dynfw.training.chunked_kl import chunked_kl_loss as chunked_kl_loss  # 分块 KL（大词表显存墙，--chunk>0 启用）


def _w(args):
    """块宽 W 统一解析：--w 优先，其次历史参数 --dla-w，否则用 block。"""
    if getattr(args, 'w', 0) > 0:
        return args.w
    if getattr(args, 'dla_w', 0) > 0:
        return args.dla_w
    return args.block


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

    elif args.arch == 'fusedfw_fw_cycle':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        m = BDHBlockFWCycleLM(D=args.dim, nh=args.nh, vocab=teacher_vocab,
                              n_layer=args.n_layer, steps=args.cycle_steps,
                              mlp_mult=args.mlp_mult, W=_w(args),
                              read_mode=getattr(args, 'fw_read', 'softmax'))

    elif args.arch == 'fusedfw_vla_cycle':
        from dynfw.models.fused_fw_vla_cycle import BDHBlockVLACycleLM
        m = BDHBlockVLACycleLM(D=args.dim, nh=args.nh, vocab=teacher_vocab,
                               n_layer=args.n_layer, steps=args.cycle_steps,
                               mlp_mult=args.mlp_mult, W=args.block)

    elif args.arch == 'fusedfw_gdn_cycle':
        from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM
        m = BDHBlockGDNCycleLM(D=args.dim, nh=args.nh, vocab=teacher_vocab,
                              n_layer=args.n_layer, steps=args.cycle_steps,
                              mlp_mult=args.mlp_mult, W=_w(args),
                              read_mode=getattr(args, 'fw_read', 'raw'))

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
        assert BDHGLA is not None, f'bdh_gla 需要 fla（flash-linear-attention）：{_GLA_ERR}'
        m = BDHGLA(D=args.dim, nh=args.nh, dk=32, vocab=teacher_vocab,
                   n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh_gla2':
        assert BDHGLAv2 is not None, f'bdh_gla2 需要 fla（flash-linear-attention）：{_GLA_ERR}'
        m = BDHGLAv2(D=args.dim, nh=args.nh, dk=args.dk, N=args.slots, vocab=teacher_vocab,
                     n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh_gla3':
        assert BDHGLAv3 is not None, f'bdh_gla3 需要 fla（flash-linear-attention）：{_GLA_ERR}'
        m = BDHGLAv3(D=args.dim, nh=args.nh, N=args.slots, vocab=teacher_vocab,
                     n_layer=args.n_layer, use_ffn=True)
    elif args.arch == 'bdh':
        m = BDHQwen(D=args.dim, n_layer=args.n_layer, nh=args.nh,
                    mlp_mult=args.mlp_mult, vocab=teacher_vocab, dropout=0.0)

    elif args.arch == 'bdh_rawfw_qwen':
        from dynfw.models.bdh_rawfw_qwen import BDHRawFWQwen
        m = BDHRawFWQwen(D=args.dim, n_layer=args.n_layer, nh=args.nh,
                         mlp_mult=args.mlp_mult, vocab=teacher_vocab, dropout=0.0,
                         W=_w(args))
    elif args.arch == 'tf':
        m = TF_sdpa(D=args.dim, nh=args.nh, n_layer=args.n_layer,
                    vocab=teacher_vocab, maxT=args.block)
    else:
        raise ValueError(f"unknown arch: {args.arch}")
    return m, m.np()


def apply_opt5(m, arch):
    """把 v6 的块内注意换成 opt5_raw 融合形态（与基线数学等价，J1 已验证）。"""
    from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw
    assert arch == 'fusedfw_fw_cycle', f'--opt5 目前仅支持 fusedfw_fw_cycle，收到 {arch}'
    to_opt5_raw(m, strict_bf16=True, bf16_prefix=False)
    cls = type(m.blocks[0].attn).__name__
    assert cls == 'FWAttentionOpt5Raw', f'替换失败: {cls}'
    print(f'=== opt5_raw 融合形态已启用（attn={cls}）===')
    return m


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
    ap.add_argument('--arch', type=str, default='fusedfw', choices=['fusedfw', 'fusedfw_cycle', 'fusedfw_rec', 'fusedfw_la', 'fusedfw_fw_cycle', 'fusedfw_vla_cycle', 'fusedfw_gdn_cycle', 'fusedfw_full', 'fusedfw_full_shared', 'fusedfw_lin', 'bdh_gla', 'bdh_gla2', 'bdh_gla3', 'bdh', 'bdh_rawfw_qwen', 'tf'],
                    help='学生架构：fusedfw / fusedfw_rec / fusedfw_la / fusedfw_full(BDH完整) / bdh / tf')
    ap.add_argument('--teacher', type=str, default='Qwen/Qwen3-0.6B')
    ap.add_argument('--data', type=str, default='',  help='语料文本文件 (每行一行)')
    ap.add_argument('--data-bin', type=str, default='', dest='data_bin',
                    help='已 tokenized 的 .bin（uint32/uint16，长 T 用）：按 --block 切块。'
                         '与 --data 二选一；必须用教师同词表的 tokenizer 生成（见 prep_corpus_qwen.py）')
    ap.add_argument('--bin-dtype', type=str, default='uint32', dest='bin_dtype',
                    choices=['uint32', 'uint16', 'int32'])
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
    ap.add_argument('--dla-k', type=int, default=16, help='DLA状态槽容量K(设小如4可触发合并)')
    ap.add_argument('--fw-read', type=str, default='raw', choices=['softmax', 'raw'],
                    dest='fw_read', help='v6 块内读侧: softmax(原版) / raw(对齐 BDH 官方)')
    ap.add_argument('--w', type=int, default=0, dest='w',
                    help='块内窗口宽 W（0=用 block）。设小(如64)可让层内出现多个 chunk，'
                         '使跨 chunk 记忆在层内累积 —— 泄漏修复后各架构必须用同一 W 才可公平对比')
    ap.add_argument('--dla-w', type=int, default=0, help='DLA块宽W(0则=block; 设小如64可在序列内多块触发合并)')
    ap.add_argument('--read-mode', type=str, default='softmaxK',
                    choices=['sum', 'softmax', 'softmaxK', 'topk'],
                    help='v8 fusedfw_slot_topk 读侧聚合: sum(原版v7无差别求和)/softmax(尺度稳定)/softmaxK(尺度对齐)/topk(稀疏)')
    ap.add_argument('--slot-topk', type=int, default=2, dest='slot_topk',
                    help='v8 read_mode=topk 时选择的槽数 k')
    ap.add_argument('--dk', type=int, default=32, help='GLA 线性注意力 head 状态宽 dk')
    ap.add_argument('--softmax', action='store_true', help='FusedFWFull 注意力加 softmax（默认 raw，消融用）')
    ap.add_argument('--tie', action='store_true', help='tie embeddings（lm_head 复用 embed，省 vocab*D 参数）')
    ap.add_argument('--teacher-half', action='store_true', dest='teacher_half',
                    help='分块路径下教师 logits 保持 fp16 常驻 GPU（落盘本就是 fp16），省一半教师显存；'
                         '块内转 fp32 计算，不改变数值口径')
    ap.add_argument('--chunk', type=int, default=0,
                    help='沿 T 分块算 KL 的块大小（>0 启用分块路径；每次只物化 chunk×V 的 logits，'
                         '实测 V=152K/B=8/T=1024 下峰值 33GiB->7GiB）。0=原全量路径')
    ap.add_argument('--snap-every', type=int, default=0,
                    help='每 N 个 step 记录一次 loss 快照（0=关闭）→ 一个 run 出整条曲线')
    ap.add_argument('--opt5', action='store_true',
                    help='v6 用 opt5_raw 融合实现（= 交付形态；与基线数学等价，J1 已验 logits maxdiff<=8.35e-07）')
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
    if args.data_bin:
        # 长 T 路径：从 tokenized .bin 切块（必须教师同词表，见 prep_corpus_qwen.py）
        import numpy as np
        dt = {'uint32': np.uint32, 'uint16': np.uint16, 'int32': np.int32}[args.bin_dtype]
        arr = np.fromfile(args.data_bin, dtype=dt)
        n = len(arr) // args.block
        if args.max_batches > 0:
            n = min(n, args.max_batches * args.batch)
        arr = arr[:n * args.block].astype(np.int64)
        ids = torch.from_numpy(arr).view(n, args.block)
        print(f"=== data tensor(bin): {ids.shape}  (T={args.block}, 块数={n}, "
              f"来源={args.data_bin}, dtype={args.bin_dtype}) ===")
    else:
        assert args.data, '必须提供 --data 或 --data-bin'
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
    if args.data_bin:
        mx = int(ids.max().item())
        assert mx < teacher_vocab, (
            f'词表不匹配！数据 token max={mx} >= 教师词表 {teacher_vocab}。'
            f'请用教师同词表 tokenizer 重新生成 .bin（experiments/distill/prep_corpus_qwen.py）')
        print(f"=== token 越界检查: max={mx} < 教师词表 {teacher_vocab}  OK ===")

    os.makedirs(args.out, exist_ok=True)
    logits_dir = os.path.join(args.out, 'teacher_logits')
    # 共享教师logits：若已缓存则直接复用，避免每次重算12.4G
    t_arr = None
    if args.shared_logits:
        shared_npy = os.path.join(args.shared_logits, 'teacher_logits.npy')
        if os.path.exists(shared_npy):
            import numpy as np
            # mmap：大 logits（如 20 块 = 50GB）不全量进 RAM（本机 125GB），逐 batch 按需读
            t_arr = np.load(shared_npy, mmap_mode='r')
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
        # 注：同配置多 seed 的 logits 完全相同 —— 批量实验请用 --shared-logits 只存一份
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
    if getattr(args, 'opt5', False):
        student = apply_opt5(student, args.arch)
    student = student.to(device)
    print(f"=== student({args.arch}) params: {n_params} ===")

    losses = []
    snaps = []
    peak_gib = 0.0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student.train()
    batch_x = torch.cat(t_blocks, dim=0)
    for epoch in range(args.epochs):
        for i in range(0, batch_x.size(0), args.batch):
            bx = batch_x[i:i+args.batch].to(device)
            _t_np = t_arr[i:i + args.batch]
            if args.chunk > 0 and args.teacher_half:
                t_lg = torch.from_numpy(_t_np).to(device)            # 保持 fp16 常驻（省一半显存）
            else:
                t_lg = torch.from_numpy(_t_np).to(device).float()
            if args.chunk > 0:
                # 分块路径：学生只产出 hidden(B,T,D)，lm_head 投影+KL 放进自定义 Function 分块做，
                # 反向按块重算 —— 不物化 B×T×V logits（实测峰值 33GiB -> 7GiB，数值等价 fp64 1e-17）
                assert hasattr(student, 'forward_hidden') and hasattr(student, 'head_params'), \
                    f"{args.arch} 未实现 forward_hidden/head_params，无法走分块路径"
                h = student.forward_hidden(bx)
                W_h, b_h = student.head_params()
                loss = chunked_kl_loss(h, W_h, b_h, t_lg, chunk=args.chunk,
                                       temperature=args.temperature)
            else:
                if args.arch in ('tf', 'bdh', 'fusedfw_full', 'fusedfw_full_shared', 'fusedfw_lin', 'bdh_gla', 'bdh_gla2', 'bdh_gla3'):
                    s_lg, _ = student.forward(bx)      # TF/BDH/FusedFWFull/FusedFWLin/BDHGLA/BDHGLAv2/BDHGLAv3 返回 (logits, loss)
                else:
                    s_lg = student.forward_logits(bx)  # FusedFW 返回 logits
                loss = kl_loss(s_lg.float(), t_lg, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if args.snap_every > 0 and len(losses) % args.snap_every == 0:
                _k = len(losses)
                _w = args.snap_every if _k >= 2 * args.snap_every else _k
                snaps.append({'step': _k, 'loss_mean_recent': float(np.mean(losses[-_w:])),
                              'loss_instant': float(losses[-1])})
            if device == 'cuda':
                peak_gib = max(peak_gib, torch.cuda.max_memory_allocated() / 2 ** 30)
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
        'dla_k': getattr(args, 'dla_k', None), 'dla_w': getattr(args, 'dla_w', None), 'block': args.block,
        'read_mode': getattr(args, 'read_mode', None), 'slot_topk': getattr(args, 'slot_topk', None),
        'cycle_steps': args.cycle_steps, 'mlp_mult': args.mlp_mult,
        'final_loss': final_loss, 'mean_loss': mean_loss,
        'ppl_est': ppl, 'wall_sec': wall, 'blocks': batch_x.size(0),
        'chunk': args.chunk, 'peak_gib': round(peak_gib, 2),
        'teacher_half': bool(args.teacher_half), 'teacher_dtype': str(t_lg.dtype),
        'snap_every': args.snap_every, 'snapshots': snaps,
        'total_steps': len(losses),
    }
    with open(os.path.join(args.out, 'distill_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    os.makedirs(os.path.join(args.out, 'checkpoint'), exist_ok=True)
    torch.save(student.state_dict(), os.path.join(args.out, 'checkpoint/student.pt'))
    print(f"=== DONE: {json.dumps(result, ensure_ascii=False)} ===")


if __name__ == '__main__':
    main()


