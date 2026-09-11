"""不变量断言 v2：head_params 投影 forward_hidden(x) == forward_logits(x)（逐 arch，绕开缺失的 fla）"""
import sys, importlib.util, torch
sys.path.insert(0, '/data/dynfw')

from dynfw.models.transformer import TF_sdpa
from dynfw.models.fused_fw_qwen import FusedFWQwen
from dynfw.models.fused_fw_cycle import FusedFWCYBLE
from dynfw.models.fused_fw_rec import FusedFWRecurrent
from dynfw.models.fused_fw_la import FusedFWLa
from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_rawfw_cycle import BDHBlockRawFWCycleLM
from dynfw.models.fused_fw_gdn_cycle import BDHBlockGDNCycleLM
from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
from dynfw.models.fused_fw_dla_topk_cycle import BDHBlockSlotCycleLM
from dynfw.models.fused_fw_full import FusedFWFull
from dynfw.models.fused_fw_full_shared import FusedFWFullShared
from dynfw.models.fused_fw_lin import FusedFWLin
from dynfw.models.bdh_qwen import BDHQwen
from dynfw.models.bdh_rawfw_qwen import BDHRawFWQwen

D, NH, MM, N, K, W, V, B, T = 32, 4, 16, 64, 4, 16, 256, 2, 32
HAS_FLA = importlib.util.find_spec('fla') is not None

builders = [
    ('tf',                   lambda: TF_sdpa(D=D, nh=NH, n_layer=1, vocab=V, maxT=T)),
    ('fusedfw',              lambda: FusedFWQwen(D=D, N=N, k=K, vocab=V, use_ffn=True, n_layer=1)),
    ('fusedfw_cycle',        lambda: FusedFWCYBLE(D=D, N=N, k=K, vocab=V, use_ffn=True, n_layer=1, steps=2)),
    ('fusedfw_rec',          lambda: FusedFWRecurrent(D=D, N=N, k=K, vocab=V, use_ffn=True, n_layer=1)),
    ('fusedfw_la',           lambda: FusedFWLa(D=D, N=N, k=K, nh=NH, vocab=V, use_ffn=True, n_layer=1)),
    ('fusedfw_la_cycle',     lambda: BDHBlockCycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM)),
    ('fusedfw_fw_cycle',     lambda: BDHBlockFWCycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM, W=W)),
    ('fusedfw_rawfw_cycle',  lambda: BDHBlockRawFWCycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM, W=W)),
    ('fusedfw_gdn_cycle',    lambda: BDHBlockGDNCycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM, W=W)),
    ('fusedfw_dla_cycle',    lambda: BDHBlockDLACycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM, W=W, K=K)),
    ('fusedfw_slot_topk',    lambda: BDHBlockSlotCycleLM(D=D, nh=NH, vocab=V, n_layer=1, steps=2, mlp_mult=MM, W=W, K=K, read_mode='softmaxK', topk=2)),
    ('fusedfw_full',         lambda: FusedFWFull(D=D, N=N, k=K, nh=NH, mlp_mult=MM, vocab=V, use_ffn=False, n_layer=1, use_softmax=False, tie=False)),
    ('fusedfw_full_shared',  lambda: FusedFWFullShared(D=D, N=N, k=K, nh=NH, mlp_mult=MM, vocab=V, use_ffn=False, n_layer=1, use_softmax=False, tie=False)),
    ('fusedfw_lin',          lambda: FusedFWLin(D=D, nh=NH, dk=32, vocab=V, n_layer=1, use_ffn=True)),
    ('bdh',                  lambda: BDHQwen(D=D, n_layer=1, nh=NH, mlp_mult=MM, vocab=V, dropout=0.0)),
    ('bdh_rawfw_qwen',       lambda: BDHRawFWQwen(D=D, n_layer=1, nh=NH, mlp_mult=MM, vocab=V, dropout=0.0, W=W)),
]

npass = nfail = 0
for name, fn in builders:
    try:
        torch.manual_seed(0)
        m = fn().cuda().eval()
        x = torch.randint(0, V, (B, T), device='cuda')
        with torch.no_grad():
            lg = m.forward_logits(x) if hasattr(m, 'forward_logits') else m.forward(x)[0]
            h = m.forward_hidden(x)
            Wt, bs = m.head_params()
            lg2 = h @ Wt.t()
            if bs is not None:
                lg2 = lg2 + bs
            d = (lg - lg2).abs().max().item()
        ok = d < 1e-4
        npass += ok; nfail += (not ok)
        print(f"{'PASS' if ok else 'FAIL'} {name:22s} hidden={str(tuple(h.shape)):14s} head={str(tuple(Wt.shape)):12s} maxdiff={d:.3e}", flush=True)
    except Exception as e:
        nfail += 1
        print(f"ERR  {name:22s} {type(e).__name__}: {str(e)[:100]}", flush=True)
    torch.cuda.empty_cache()

# fla 依赖的 GLA 系列：库缺失则如实标注
if HAS_FLA:
    from dynfw.models.bdh_gla import BDHGLA
    from dynfw.models.bdh_gla_v2 import BDHGLAv2
    from dynfw.models.bdh_gla_v3 import BDHGLAv3
    for name, fn in [('bdh_gla', lambda: BDHGLA(D=D, nh=NH, dk=32, vocab=V, n_layer=1, use_ffn=True)),
                     ('bdh_gla2', lambda: BDHGLAv2(D=D, nh=NH, dk=8, N=N, vocab=V, n_layer=1, use_ffn=True)),
                     ('bdh_gla3', lambda: BDHGLAv3(D=D, nh=NH, N=N, vocab=V, n_layer=1, use_ffn=True))]:
        try:
            m = fn().cuda().eval()
            x = torch.randint(0, V, (B, T), device='cuda')
            with torch.no_grad():
                lg = m.forward_logits(x); h = m.forward_hidden(x); Wt, bs = m.head_params()
                lg2 = h @ Wt.t() + (bs if bs is not None else 0)
                d = (lg - lg2).abs().max().item()
            ok = d < 1e-4; npass += ok; nfail += (not ok)
            print(f"{'PASS' if ok else 'FAIL'} {name:22s} maxdiff={d:.3e}", flush=True)
        except Exception as e:
            nfail += 1
            print(f"ERR  {name:22s} {type(e).__name__}: {str(e)[:100]}", flush=True)
else:
    print("SKIP bdh_gla/bdh_gla2/bdh_gla3  —— 环境缺 fla（flash-linear-attention），接口已注入但未跑断言")

print(f"\n结果: {npass} 通过 / {nfail} 未通过")
