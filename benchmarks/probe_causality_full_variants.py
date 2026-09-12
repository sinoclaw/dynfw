"""补验：台账「最优」FusedFWFull / 省参最优 FusedFWFullShared 的因果性。
原 probe_causality_all.py 只覆盖 10 个架构，未含这两个（批1 冠军），按「缺陷无存量豁免」须实证。
"""
import torch
from dynfw.models.fused_fw_full import FusedFWFull
from dynfw.models.fused_fw_full_shared import FusedFWFullShared

DEV = 'cuda'
torch.manual_seed(0)
POS = 48

def probe(name, mk):
    m = mk().to(DEV).eval()
    T = POS + 2
    x = torch.randint(0, 1000, (1, T), device=DEV)
    with torch.no_grad():
        a = m(x[:, :POS + 1])[0]
        b = m(x[:, :POS + 2])[0]
    d = (a[0, POS] - b[0, POS]).abs().max().item()
    verdict = 'CAUSAL OK' if d < 1e-5 else 'LEAK!!'
    print(f'{name:48s} maxdiff={d:.3e}  {verdict}', flush=True)
    del m
    torch.cuda.empty_cache()

print('=== 补验：批1 冠军/省参冠军的因果性（长度依赖探针 POS=%d）===' % POS)
probe('FusedFWFull L=3 (批1 最优 KL 93.23)', lambda: FusedFWFull(D=64, N=512, k=16, nh=4, mlp_mult=16, vocab=1000, use_ffn=False, n_layer=3))
probe('FusedFWFull L=1 (对照/单层)', lambda: FusedFWFull(D=64, N=512, k=16, nh=4, mlp_mult=16, vocab=1000, use_ffn=False, n_layer=1))
probe('FusedFWFullShared L=3 (省参最优 115.24)', lambda: FusedFWFullShared(D=64, N=512, k=16, nh=4, mlp_mult=16, vocab=1000, use_ffn=False, n_layer=3))
probe('FusedFWFull+softmax L=3 (消融变体)', lambda: FusedFWFull(D=64, N=512, k=16, nh=4, mlp_mult=16, vocab=1000, use_ffn=False, n_layer=3, use_softmax=True))
print('=== 补验结束 ===')
