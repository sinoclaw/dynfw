"""静态形状解码器（DynDecoderS / CUDA Graph 版）的正确性闸门。

闸门A: fp32、两边同路径       → 只该剩归约顺序噪声（须 <1e-5）
闸门B: bf16 部署口径          → 报相对差 + top-1 一致率（与 TF 地板同阶即可）
闸门C: CUDA Graph 版 vs eager → 必须逐位一致（防「位置被烘进图」这类错）
用法: PYTHONPATH=/data/dynfw python benchmarks/check_decs.py
"""
import sys
import torch

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from benchmarks.bench_decode import DynDecoderS, VOCAB, D, NH, NL, W

T = 512
NREP = 4


def build():
    torch.manual_seed(0)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL,
                             steps=1, mlp_mult=4, W=W).cuda().eval()


def main():
    torch.manual_seed(3)
    seq = torch.randint(0, VOCAB, (1, T + 8), device='cuda')

    # ---- 闸门 A / B ----
    for tag, amp, dt in (('A fp32', False, torch.float32), ('B bf16', True, torch.bfloat16)):
        lm = build()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                ref = lm(seq[:, :T + NREP], None)[0][0].float()[T:T + NREP]
                ds = DynDecoderS(lm, B=1, dt=dt, amp=amp)
                ds.prefill(seq[:, :T])
                got = []
                for k in range(T, T + NREP):
                    ds.tokbuf[:, 0] = seq[:, k]
                    ds.step()
                    got.append(ds.outbuf[:, -1, :].float().clone())
                    ds.advance()
                got = torch.cat(got, 0)
        md = (ref - got).abs().max().item(); sc = ref.abs().max().item()
        t1 = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
        ok = 'PASS' if md / sc < (1e-5 if not amp else 1e-2) else 'CHECK'
        print(f'  闸门{tag:<8} maxdiff={md:.4e} 相对={md/sc:.3e} (幅值{sc:.2f}) '
              f'top-1={t1*100:.1f}%  {ok}')

    # ---- 闸门 C：CUDA Graph vs eager ----
    lm = build()
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ds_e = DynDecoderS(lm, B=1); ds_e.prefill(seq[:, :T])
            e = []
            for k in range(T, T + NREP):
                ds_e.tokbuf[:, 0] = seq[:, k]
                ds_e.step(); e.append(ds_e.outbuf.clone()); ds_e.advance()
            e = torch.cat([t[:, -1, :].float() for t in e], 0)

            ds_g = DynDecoderS(lm, B=1); ds_g.prefill(seq[:, :T])
            ds_g.tokbuf[:, 0] = seq[:, T]
            for _ in range(3):
                ds_g.step(); ds_g.advance()
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                ds_g.step()
            # ⚠️ 捕获后必须【把状态重置回 pos=T】再比 —— 否则是拿「已推进3步」的去比「清空状态」的
            #    （prefill 已是原地拷贝，重置不会打断图的地址绑定）
            ds_g.prefill(seq[:, :T])
            g = []
            for k in range(T, T + NREP):
                ds_g.tokbuf[:, 0] = seq[:, k]
                cg.replay()
                g.append(ds_g.outbuf.clone())
                ds_g.advance()
            g = torch.cat([t[:, -1, :].float() for t in g], 0)
    md = (e - g).abs().max().item(); sc = e.abs().max().item()
    print(f'  闸门C graph     maxdiff={md:.4e} 相对={md/sc:.3e}  '
          f'{"PASS 图与 eager 逐位一致" if md < 1e-3 else "FAIL 图里烘死了常量!"}')
    print('  （C 若 FAIL，说明 rope 位置/状态更新被烘进图 → 那个性能数字是错计算的耗时，不能报）')


if __name__ == '__main__':
    main()
