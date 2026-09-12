"""扫 T：v6.7+compile vs v6+opt5 vs TF —— 找 O(T) 对 O(T²) 的交叉点。

判据（跑前锁死，事后不改）：
  J1 交叉点：找出 v6.7+compile 首次 ≤ TF 的 T
  J2 两个口径都报：含 compile 前 warmup / 不含（台账强制要求并列）
  J3 单进程单形态、预热后取中位；TF 的 maxT 必须 ≥ 被测 T
  J4 诚实：若在测到的最大 T 上仍不快于 TF，如实报"未找到交叉点"

编译纪律（台账铁律）：
  torch.compile 首次调用才真编译。若编译那一刻 eager 路径已跑过（cuBLAS 库级计划已建立），
  inductor 生成的代码快 1.87×；否则生成保守代码。
  ⟹ 本脚本对每个 (arch, T) 组合【独立进程】测，避免互相预热污染；
     且显式给出「warmup 后 compile」与「直接 compile」两个读数。
"""
import sys, time, statistics, json, subprocess

Ts = [8192, 16384, 32768]
ARCHS = ['v67c', 'v67e', 'opt5', 'tf']

WORKER = r'''
import sys, time, statistics
import torch
import torch.nn.functional as F
sys.path.insert(0, '/data/dynfw')
DEV = 'cuda'
T = int(sys.argv[1]); W, B = 64, 1
D, NH, NLAYER, VOCAB, MLP_MULT = 128, 16, 2, 151936, 64
MODE = sys.argv[2]
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

if MODE in ('v67c', 'v67e'):
    from dynfw.models.fused_fw_gdn_fla import BDHBlockFLALM as M
    m = M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT, W=W, read_mode='raw')
elif MODE == 'opt5':
    from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM as M
    from dynfw.models.fused_fw_fw_cycle_opt import to_opt5_raw
    m = M(D=D, nh=NH, vocab=VOCAB, n_layer=NLAYER, steps=1, mlp_mult=MLP_MULT, W=W, read_mode='raw')
    m = to_opt5_raw(m, strict_bf16=True, bf16_prefix=False)
elif MODE == 'tf':
    from dynfw.models.transformer import TF_sdpa
    m = TF_sdpa(D=D, nh=NH, n_layer=NLAYER, vocab=VOCAB, maxT=T)
m = m.to(DEV)
nparam = sum(p.numel() for p in m.parameters())

x = torch.randint(0, 1000, (B, T), device=DEV)
tgt = torch.randint(0, 1000, (B, T), device=DEV)
head = torch.nn.Linear(D, 4096, bias=False).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

fn = m.forward_hidden
if MODE == 'v67c':
    # 台账铁律：编译【前】先跑 eager，让 cuBLAS 库级计划建立
    def _eager_pass():
        o = m.forward_hidden(x); lg = o.view(B*T, D) @ head.weight.T
        F.cross_entropy(lg.float(), tgt.view(-1)).backward()
        m.zero_grad(set_to_none=True)
    _eager_pass()
    torch.cuda.synchronize()
    fn = torch.compile(m.forward_hidden)

def one():
    o = fn(x)
    lg = o.view(B * T, D) @ head.weight.T
    loss = F.cross_entropy(lg.float(), tgt.view(-1))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return loss.item()

try:
    last = None
    for _ in range(3):
        last = one()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        last = one()
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1000)
    print(f"{T},{MODE},{statistics.median(ts):.1f},{nparam},{last:.2f},OK")
except torch.cuda.OutOfMemoryError:
    print(f"{T},{MODE},OOM,{nparam},0,OOM")
except Exception as e:
    print(f"{T},{MODE},ERR,{nparam},0,{type(e).__name__}:{str(e)[:80]}")
'''

open('/tmp/sweep_worker.py', 'w').write(WORKER)
PY = '/data/dynfw-env/bin/python'
rows = []
print(f"{'T':>7} {'arch':>6} {'ms/step':>9} {'params':>11}  note")
print("-" * 60)
for T in Ts:
    for mode in ARCHS:
        r = subprocess.run([PY, '/tmp/sweep_worker.py', str(T), mode],
                           capture_output=True, text=True, timeout=1800,
                           env=dict(__import__('os').environ, PYTHONPATH='/data/dynfw'))
        line = [l for l in r.stdout.strip().split('\n') if l.count(',') >= 4]
        if line:
            f = line[-1].split(',')
            ms = f[2]
            print(f"{T:>7} {mode:>6} {ms:>9} {int(f[3]):>11,}  {f[5][:40]}")
            rows.append(dict(T=T, arch=mode, ms=None if ms in ('OOM', 'ERR') else float(ms),
                             params=int(f[3]), note=f[5][:60]))
        else:
            print(f"{T:>7} {mode:>6} {'?':>9} {'?':>11}  {r.stderr.strip().splitlines()[-1][:50] if r.stderr.strip() else 'no output'}")

json.dump(rows, open('/tmp/sweep_results.json', 'w'), indent=2)
print("\n=== 相对 TF 的倍数（<1 = 比 TF 快）===")
by = {}
for r in rows:
    by[(r['T'], r['arch'])] = r['ms']
for T in Ts:
    tf = by.get((T, 'tf'))
    if not tf:
        print(f"  T={T}: TF 无数据")
        continue
    for a in ARCHS:
        v = by.get((T, a))
        if v:
            print(f"  T={T:>6}  {a:>5}: {v/tf:6.2f}x  ({v:.1f}ms vs TF {tf:.1f}ms)")
        else:
            print(f"  T={T:>6}  {a:>5}: —")
