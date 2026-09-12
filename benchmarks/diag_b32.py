"""诊断：B=32 为什么还输 —— 逐项消融静态版的额外开销，并给出修法。

假设：静态版每步都算完整 fold（fp32 einsum + 两次 .float() 上采样），但 256 步只有 1 步需要。
      这是「消掉 if 让图能捕获」的代价，B=32 时被放大 32 倍。
修法：捕获【两张图】（块边界版 / 非边界版），运行时按 pos 选 → 非边界步完全不碰 fold。

用法: PYTHONPATH=/data/dynfw python benchmarks/diag_b32.py --batch 32 --T 8192
"""
import sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import FWAttentionOpt
from benchmarks.bench_decode import DynDecoderS, TFDecoder, build_tf, VOCAB, D, NH, NL, W


def build(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL,
                             steps=1, mlp_mult=4, W=W).cuda().eval()


# ---- 消融版：fold_dtype / skip_fold / skip_where 三个开关 ----
class DynAbl(DynDecoderS):
    fold_dtype = torch.float32    # 折叠用什么 dtype 累加
    skip_fold = False             # True = 完全不算 fold（错的，仅诊断用）
    skip_where = False            # True = 不写窗口（错的，仅诊断用）

    def _blk(self, blk, s, x):
        C = blk.config; D_ = blk.D; nh = C.n_head
        N = C.mlp_internal_dim_multiplier * D_ // nh
        B = x.shape[0]; W_ = self.W
        x = blk.ln(x)
        for _ in range(blk.steps):
            x_latent = x @ blk.encoder
            x_sparse = F.relu(x_latent)
            r = self.posbuf.float().view(1, 1, 1, 1)
            QR = FWAttentionOpt.rope_fast(r * blk.attn.freqs, x_sparse)
            if not self.skip_fold:
                if self.fold_dtype == torch.float32:
                    fold = torch.einsum('bhmd,bme->bhde', s['kw'].float(), s['vw'][:, 0].float())
                else:
                    fold = torch.einsum('bhmd,bme->bhde', s['kw'], s['vw'][:, 0]).float()
                s['mem'].add_(self.isb * fold)
            oh = (self.ar == self.slot).view(1, 1, W_, 1)
            if not self.skip_where:
                s['kw'].copy_(torch.where(oh, QR, s['kw']))
                s['vw'].copy_(torch.where(oh, x, s['vw']))
            sim = torch.einsum('bhd,bhmd->bhm', QR[:, :, 0], s['kw'])
            keep = (self.ar <= self.slot).view(1, 1, W_)
            sim = sim.masked_fill(~keep, float('-inf'))
            p = torch.softmax(sim.float(), dim=-1)
            agg = torch.einsum('bhm,bme->bhe', p, s['vw'][:, 0])
            retr = torch.einsum('bhd,bhde->bhe', QR[:, :, 0], s['mem'])
            yKV = blk.ln((agg + retr).unsqueeze(2))
            y_latent = yKV @ blk.encoder_v
            y_sparse = F.relu(y_latent)
            xy = x_sparse * y_sparse
            yMLP = xy.transpose(1, 2).reshape(B, 1, 1, N * nh) @ blk.decoder
            x = blk.ln(x + blk.ln(yMLP))
        return x


# ---- 修法：两张图（边界版含 fold / 非边界版不含）----
class DynDecoderS2(DynDecoderS):
    do_fold = True

    def _blk(self, blk, s, x):
        C = blk.config; D_ = blk.D; nh = C.n_head
        N = C.mlp_internal_dim_multiplier * D_ // nh
        B = x.shape[0]; W_ = self.W
        x = blk.ln(x)
        for _ in range(blk.steps):
            x_latent = x @ blk.encoder
            x_sparse = F.relu(x_latent)
            r = self.posbuf.float().view(1, 1, 1, 1)
            QR = FWAttentionOpt.rope_fast(r * blk.attn.freqs, x_sparse)
            if self.do_fold:      # Python 层分支 → 捕获时烘死，但两张图分别捕获
                fold = torch.einsum('bhmd,bme->bhde', s['kw'].float(), s['vw'][:, 0].float())
                s['mem'].add_(fold)
            oh = (self.ar == self.slot).view(1, 1, W_, 1)
            s['kw'].copy_(torch.where(oh, QR, s['kw']))
            s['vw'].copy_(torch.where(oh, x, s['vw']))
            sim = torch.einsum('bhd,bhmd->bhm', QR[:, :, 0], s['kw'])
            keep = (self.ar <= self.slot).view(1, 1, W_)
            sim = sim.masked_fill(~keep, float('-inf'))
            p = torch.softmax(sim.float(), dim=-1)
            agg = torch.einsum('bhm,bme->bhe', p, s['vw'][:, 0])
            retr = torch.einsum('bhd,bhde->bhe', QR[:, :, 0], s['mem'])
            yKV = blk.ln((agg + retr).unsqueeze(2))
            y_latent = yKV @ blk.encoder_v
            y_sparse = F.relu(y_latent)
            xy = x_sparse * y_sparse
            yMLP = xy.transpose(1, 2).reshape(B, 1, 1, N * nh) @ blk.decoder
            x = blk.ln(x + blk.ln(yMLP))
        return x


def median(fn, reps=7, warm=2):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            t0 = time.time(); fn(); torch.cuda.synchronize()
            ts.append(1000 * (time.time() - t0))
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--T', type=int, default=8192)
    a = ap.parse_args()
    B, T = a.batch, a.T
    torch.manual_seed(0)
    seq = torch.randint(0, VOCAB, (B, T + 64), device='cuda')
    p = T      # 从 T 开始解码（T 是 W 的倍数 → 刚好在块边界）

    print('=' * 92)
    print(f'诊断 batch={B} T={T}  —— 「为什么 B=32 还输」')
    print('=' * 92)

    def prep(cls, **kw):
        m = cls(build(), B=B)                     # 消融开关是类属性，实例上赋值
        for k, v in kw.items():
            setattr(m, k, v)
        m.prefill(seq[:, :T]); m.tokbuf[:, 0] = seq[:, T]
        return m

    # 1) 逐项消融
    abl = [
        ('① 静态版 现状(每步fp32 fold+where)', dict(), ),
        ('② fold 改 bf16（省两次 .float()）', dict(fold_dtype=torch.bfloat16)),
        ('③ 完全不算 fold（错的·仅诊断）', dict(skip_fold=True)),
        ('④ 不写窗口（错的·仅诊断）', dict(skip_where=True)),
        ('⑤ ③+④ 都去掉（下界·错的）', dict(skip_fold=True, skip_where=True)),
    ]
    for name, kw in abl:
        m = prep(DynAbl, **kw)
        ms = median(lambda: (m.step(), m.advance()))
        print(f'  {name:<36} {ms:7.2f} ms/step')
        del m; torch.cuda.empty_cache()

    # 2) 修法：两张图
    print()
    m = prep(DynDecoderS2)
    with torch.no_grad():
        m.do_fold = False
        for _ in range(3):
            m.step(); m.advance()
        g_no = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_no):
            m.step()
        m.do_fold = True
        m.prefill(seq[:, :T]); m.tokbuf[:, 0] = seq[:, T]
        for _ in range(3):
            m.step(); m.advance()
        g_fold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fold):
            m.step()

        # 闸门C：两张图联手必须与 eager 逐位一致
        m.prefill(seq[:, :T])
        ref = []
        for k in range(4):
            pos = T + k
            # ⚠️ 参照也必须按 pos 切 do_fold —— 否则参照每步都 fold（只有块边界那步该 fold）
            m.do_fold = (pos % W == 0 and pos > 0)
            m.tokbuf[:, 0] = seq[:, pos]; m.step(); ref.append(m.outbuf.clone()); m.advance()
        m.prefill(seq[:, :T]); m.do_fold = True
        got = []
        for k in range(4):
            pos = T + k
            m.tokbuf[:, 0] = seq[:, pos]
            (g_fold if (pos % W == 0 and pos > 0) else g_no).replay()
            got.append(m.outbuf.clone()); m.advance()
        md = max((a_-b_).abs().max().item() for a_, b_ in zip(ref, got))
        print(f'  【修法】两张图 vs eager 闸门C maxdiff = {md:.3e}  '
              f'{"PASS" if md < 1e-3 else "FAIL"}')

        # 计时：按 pos 选图（真实解码路径）
        def run_two():
            pos = m.pos
            (g_fold if (pos % W == 0 and pos > 0) else g_no).replay()
            m.advance()
        m.prefill(seq[:, :T]); m.do_fold = True
        ms2 = median(run_two)
        print(f'  【修法】两张图分流                {ms2:7.2f} ms/step')

    # 3) 对手同场
    print()
    tfd = TFDecoder(build_tf(T + 64), B=B, maxlen=T + 64); tfd.prefill(seq[:, :T])
    tf_ms = median(lambda: tfd.step(seq[:, tfd.pos]))
    print(f'  TF 原生动态(同场)                 {tf_ms:7.2f} ms/step')
    print()
    print(f'  → 修法后 我们/TF = {tf_ms/ms2:.2f}x  （>1 = 我们更快）')


if __name__ == '__main__':
    main()
