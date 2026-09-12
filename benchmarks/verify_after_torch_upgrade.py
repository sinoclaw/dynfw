"""升级后验收（判据 J1-J6，跑前已锁死，事后不改）。

J1 关键：v6.7 + compile 不再抛 Autotuner 异常（能出图）
J2 数值：v6.7 的 loss 与参考值一致（参考：升级前 1000step 长T 中位 5738.0；此处用短冒烟相对比较）
J3 速度：compile 后 v6.7 应 < 70.5ms；目标 ≤ 56.9ms（打平 v6+opt5）
J4 FLA 冒烟仍通过
J5 v6 / v6+opt5 读数不变
J6 回滚就绪：快照文件存在
"""
import sys, os, time, statistics, json
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
sys.setrecursionlimit(10000)
DEV = 'cuda'
T, W, B = 8192, 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
torch.backends.cuda.matmul.allow_tf32 = True
R = {}

print("=" * 88)
print("环境")
print("=" * 88)
import triton
print(f"  torch {torch.__version__} (cuda {torch.version.cuda}) | triton {triton.__version__}")

print()
print("=" * 88)
print("J4: FLA 冒烟")
print("=" * 88)
try:
    from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
    import torch.nn.functional as FF
    Bq, Tq, H, K, V = 1, 512, 4, 64, 64
    q = torch.randn(Bq, Tq, H, K, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(Bq, Tq, H, K, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(Bq, Tq, H, V, device=DEV, dtype=torch.bfloat16)
    g = FF.logsigmoid(torch.randn(Bq, Tq, H, device=DEV, dtype=torch.bfloat16))
    o, st = fused_chunk_simple_gla(q, k, v, g, output_final_state=True)
    print(f"  J4 ✓  FLA OK  o={tuple(o.shape)} state={tuple(st.shape)}")
    R['J4'] = True
except Exception as e:
    print(f"  J4 ✗  {type(e).__name__}: {str(e)[:200]}")
    R['J4'] = False


def build(which):
    if which == 'v67':
        from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM
        return BDHBlockFLALM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                             mlp_mult=MLP_MULT, W=W, read_mode='raw').to(DEV)
    if which == 'v6':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                                 mlp_mult=MLP_MULT, W=W, read_mode='raw').to(DEV)
    if which == 'opt5':
        from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
        from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw
        m = BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1,
                              mlp_mult=MLP_MULT, W=W, read_mode='raw')
        return to_opt5_raw(m, strict_bf16=True, bf16_prefix=False).to(DEV)


x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)


def bench(m, label, mode='eager', iters=3, warm=2):
    """mode: eager | compile"""
    torch.manual_seed(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    fn = m.forward_hidden
    if mode == 'compile':
        try:
            fn = torch.compile(m.forward_hidden)
        except Exception as e:
            print(f"  {label:32s} compile 注册失败 {type(e).__name__}")
            return None, None

    def one():
        o = fn(x)
        lg = o.view(B * T, D) @ head.weight.T
        loss = F.cross_entropy(lg.float(), tgt.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        return loss.item()
    try:
        last = None
        for _ in range(warm):
            last = one()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            last = one()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(ts)
        print(f"  {label:32s} {ms:8.1f} ms   loss={last:.1f}")
        return ms, last
    except Exception as e:
        msg = str(e)[:130]
        flag = 'Autotuner' in msg or 'reset_idx' in msg
        print(f"  {label:32s} FAIL{'(Autotuner!)' if flag else ''} {type(e).__name__}: {msg}")
        return None, None


print()
print("=" * 88)
print("J1 + J3: v6.7 在 compile 下能否出图 / 速度")
print("=" * 88)
m = build('v67')
ms_e, l_e = bench(m, 'v6.7 eager')
del m; torch.cuda.empty_cache()
m = build('v67')
ms_c, l_c = bench(m, 'v6.7 compile', mode='compile')
R['J1'] = ms_c is not None
R['J3_v67_eager'], R['J3_v67_compile'] = ms_e, ms_c
del m; torch.cuda.empty_cache()

print()
print("=" * 88)
print("J5: v6 / v6+opt5 读数不变（参照：v6≈953ms, opt5≈56.9ms）")
print("=" * 88)
m = build('v6'); ms_v6, _ = bench(m, 'v6 eager'); del m; torch.cuda.empty_cache()
m = build('opt5'); ms_o, _ = bench(m, 'v6+opt5 eager'); del m; torch.cuda.empty_cache()
R['J5_v6'], R['J5_opt5'] = ms_v6, ms_o

print()
print("=" * 88)
print("验收汇总")
print("=" * 88)
print(f"  J1 v6.7+compile 出图:      {'✓' if R.get('J1') else '✗'}")
print(f"  J3 v6.7 eager/compile:     {R.get('J3_v67_eager')} / {R.get('J3_v67_compile')} ms")
print(f"  J4 FLA 冒烟:               {'✓' if R.get('J4') else '✗'}")
print(f"  J5 v6={R.get('J5_v6')} opt5={R.get('J5_opt5')} ms（参照 953 / 56.9）")
print(f"  J6 快照:                   {'✓' if os.path.exists('/data/dynfw/docs/env-snapshot-pre-torch-upgrade-20260913.txt') else '✗'}")
json.dump(R, open('/data/dynfw/results/verify_after_torch_upgrade.json', 'w'), indent=2, default=str)
print("\n  结果已存 results/verify_after_torch_upgrade.json")
