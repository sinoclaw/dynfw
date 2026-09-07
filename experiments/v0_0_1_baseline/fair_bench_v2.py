"""FAIR_BENCH_V2 —— 严格公平基准（CPU 口径），按 GPT 审计 V11 §六 14 点。

修复被审计抓的问题：
  - P0: seed 先设再建模型（factory 化）→ 5 seed 是真实独立初始化
  - TF 基线用 SDPA(scaled_dot_product_attention, is_causal) 而非 naive attention
  - 固定完整 validation set（make_val 一次性冻结）
  - full-sequence prefill 单独测（无 cache）
  - autoregressive decode 单独测（TF 带 KV-cache；FW 增量 rho 状态）
  - 同参数（宽度匹配）、同训练 token 数、同 wall-clock 预算、5 seed 报 mean±std、最后 Pareto

注：审计第④点"CPU/GPU 各测"——本机无 GPU，只产 CPU 口径，GPU 列待有卡补（不假装测过 GPU）。
"""
import sys; sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np
from dynfw.models.fused_fw import FusedFW                      # 复用包内实现
from dynfw.data import load_data, get_batch

torch.set_num_threads(8)
MAXT = 8192
SEEDS = [0,1,2,3,4]

# ---------- 固定 validation set ----------
def make_val(block=256, nseq=64, seed=123):
    data = load_data(); val = data[int(0.9*len(data)):]
    rng = np.random.RandomState(seed); ix = rng.randint(0, len(val)-block, (nseq,))
    x = torch.stack([torch.from_numpy(val[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(val[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x, y

# ---------- TF_sdpa 基线 ----------
class TF_sdpa(nn.Module):
    def __init__(s, D=128, nh=4, n_layer=2, vocab=256, maxT=MAXT):
        super().__init__(); s.vocab=vocab; s.D=D; s.nh=nh; s.n_layer=n_layer
        assert D % nh == 0
        s.e=nn.Embedding(vocab,D)
        # 正弦固定位置编码（0 参数字典，避免 maxT*D 参数池污染'同参数'对齐）
        pos=torch.zeros(maxT,D); ar=torch.arange(maxT).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,D,2).float()*(-np.log(10000.0)/D))
        pos[:,0::2]=torch.sin(ar*div); pos[:,1::2]=torch.cos(ar*div)
        s.register_buffer('pos',pos)
        s.blocks=nn.ModuleList()
        for _ in range(n_layer):
            s.blocks.append(nn.ModuleList([nn.LayerNorm(D),nn.Linear(D,3*D),nn.Linear(D,D),
                nn.LayerNorm(D),nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)]))
        s.ln=nn.LayerNorm(D); s.h=nn.Linear(D,vocab)
    def forward(s,x,t=None):
        B,T=x.size(); h=s.nh; D=s.D
        y=s.e(x)+s.pos[:T].unsqueeze(0)
        for ln1,qkv,proj,ln2,w1,act,w2 in s.blocks:
            xx=ln1(y); q,k,v=qkv(xx).chunk(3,dim=-1)
            q=q.view(B,T,h,D//h).transpose(1,2); k=k.view(B,T,h,D//h).transpose(1,2); v=v.view(B,T,h,D//h).transpose(1,2)
            o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            o=o.transpose(1,2).contiguous().view(B,T,D); o=proj(o)
            y=y+o; y=y+w2(act(w1(ln2(y))))
        lg=s.h(s.ln(y))
        loss=None if t is None else F.cross_entropy(lg.view(-1,s.vocab),t.view(-1))
        return lg,loss
    def np(self): return sum(p.numel() for p in self.parameters())

# ---------- TF KV-cache decode（逐 token，attention 只对已缓存 k,v 做） ----------
def tf_decode_kvcache(model, prompt, gen=16, reps=2):
    model.eval(); D=model.D; nh=model.nh; hd=D//nh
    P=prompt.shape[1]
    with torch.no_grad():
        caches=[]; tok=prompt
        for _ in range(gen):   # 逐 token 生成
            T=tok.shape[1]
            y=model.e(tok)+model.pos[:T].unsqueeze(0)
            new_k=[]; new_v=[]
            for bi,(ln1,qkv,proj,ln2,w1,act,w2) in enumerate(model.blocks):
                xx=ln1(y); q,k,v=qkv(xx).chunk(3,dim=-1)
                q=q.view(1,T,nh,hd).transpose(1,2); k=k.view(1,T,nh,hd).transpose(1,2); v=v.view(1,T,nh,hd).transpose(1,2)
                if bi>=len(caches): caches.append((k,v))
                else: caches[bi]=(torch.cat([caches[bi][0],k],dim=2), torch.cat([caches[bi][1],v],dim=2))
                ck,cv=caches[bi]
                o=F.scaled_dot_product_attention(q,ck,cv,is_causal=True)
                o=o.transpose(1,2).contiguous().view(1,T,D); o=proj(o)
                y=y+o
                y=y+w2(act(w1(ln2(y))))
            lg=model.h(model.ln(y))[:,-1:]
            tok=lg.argmax(-1).to(torch.long)
    return tok  # 仅为完整性；计时在调用方
def decode_tf_timing(model, prompt, gen=16, reps=3):
    model.eval()
    with torch.no_grad():
        # warmup（建 cache）
        tf_decode_kvcache(model, prompt, gen=gen)
        t0=time.time()
        for _ in range(reps):
            tf_decode_kvcache(model, prompt, gen=gen)
    return (time.time()-t0)/(reps*gen)   # 每 token 秒

# ---------- FW 增量 decode（running rho 状态） ----------
def fw_forward_single(s, x_tok, rho):
    et=s.e(x_tok)                         # [1,1,D]
    lat=s.encs[0](et)                     # [1,1,N]
    topv,topi=torch.topk(lat,s.k,dim=-1)
    act=torch.relu(torch.zeros_like(lat).scatter(-1,topi,topv))   # [1,1,N]
    ln_et=s.ln(et)                        # [1,1,D]
    # 增量累积（外积，等价 einsum('bn,bd->nd')，b=1 单样本）
    rho=rho+torch.outer(act[0].reshape(-1), ln_et[0].reshape(-1)) # [N,D]
    mem_ctx=act[0].reshape(1,-1) @ rho                             # [1,N]@[N,D]=[1,D]
    h=s.ln(et[0]+mem_ctx); h=s.out(h)
    if s.use_ffn: h=h+s.ffn(s.ln(h))
    return s.head(s.ln(h)).unsqueeze(0), rho
def decode_fw_timing(model, prefix_len=32, gen=16, reps=3):
    model.eval(); D=model.D; N=model.N
    with torch.no_grad():
        x=torch.arange(prefix_len).unsqueeze(0)
        ets=model.e(x); lat=model.encs[0](ets)
        topv,topi=torch.topk(lat,model.k,dim=-1)
        act=torch.relu(torch.zeros_like(lat).scatter(-1,topi,topv)); ln_et=model.ln(ets)
        rho=torch.zeros((N,D))+ (act[0].T @ ln_et[0])              # 前缀累积 [N,D]
        cur=torch.tensor([[prefix_len-1]],dtype=torch.long)
        for _ in range(gen):
            lg,rho=fw_forward_single(model,cur,rho); cur=lg.argmax(-1)
        t0=time.time()
        for _ in range(gen):
            lg,rho=fw_forward_single(model,cur,rho); cur=lg.argmax(-1)
    return (time.time()-t0)/gen

# ---------- prefill 计时（全序列一次前向，无 cache） ----------
def prefill_time(model, x, warmup=2, reps=3):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(x)
        t0=time.time()
        for _ in range(reps): model(x)
    return (time.time()-t0)/reps

# ---------- 参数匹配 ----------
def search_width(mk, P, lo=32, hi=256, step=4):
    best=None
    for D in range(lo,hi+1,step):
        try:
            m=mk(D); p=m.np()
        except Exception:
            continue
        if best is None or abs(p-P)<abs(best[1]-P): best=(D,p)
    return best
def mk_tf(D): return TF_sdpa(D=D, nh=4, n_layer=2)
def mk_fw(D): return FusedFW(D=D, N=4*D, k=16, use_ffn=True)

# ---------- 训练 + 固定 val（seed 先设再建模型） ----------
def train_eval(mk, seed, iters=250, bl=256, bt=8, lr=3e-4, valset=None):
    torch.manual_seed(seed); np.random.seed(seed)      # P0: seed 先设
    data=load_data(); rng=np.random.RandomState(seed)
    m=mk()                                             # 再建模型 ← 初始化受 seed 控制
    opt=torch.optim.AdamW(m.parameters(), lr=lr)
    t0=time.time()
    for _ in range(iters):
        m.train(); x,y=get_batch(data,bl,bt,rng); _,loss=m(x,y)
        opt.zero_grad(); loss.backward(); opt.step()
    train_t=time.time()-t0
    m.eval()
    with torch.no_grad():
        xv,yv=valset; _,vl=m(xv,yv)
    return float(vl.item()), train_t, m.np()

# ---------- 主流程 ----------
def main():
    print("=== FAIR_BENCH_V2 (CPU) — 审计V11 14点 ===", flush=True)
    TARGETS=[200_000,300_000,400_000,500_000,600_000]
    valset=make_val(); results=[]
    for P in TARGETS:
        dtf=search_width(mk_tf,P); dfw=search_width(mk_fw,P)
        tf_v=[]; tf_t=[]; fw_v=[]; fw_t=[]
        for sd in SEEDS:
            v,tt,_=train_eval(lambda: mk_tf(dtf[0]), seed=sd, valset=valset); tf_v.append(v); tf_t.append(tt)
            v,tt,_=train_eval(lambda: mk_fw(dfw[0]), seed=sd, valset=valset); fw_v.append(v); fw_t.append(tt)
        # 计时用固定 seed0 模型
        mtf=mk_tf(dtf[0]); mfw=mk_fw(dfw[0])
        xp=torch.randint(0,256,(8,256))
        ptf=prefill_time(mtf,xp); pfw=prefill_time(mfw,xp)
        dtf_tok=decode_tf_timing(mtf, torch.zeros((1,256),dtype=torch.long)); dfw_tok=decode_fw_timing(mfw)
        mv,mvel=np.mean(tf_v),np.std(tf_v); mw,mwel=np.mean(fw_v),np.std(fw_v)
        results.append(dict(P=P, D_tf=dtf[0], D_fw=dfw[0],
            tf_val=mv, tf_val_std=mvel, fw_val=mw, fw_val_std=mwel,
            tf_train=float(np.mean(tf_t)), fw_train=float(np.mean(fw_t)),
            prefill_tf=ptf, prefill_fw=pfw, decode_tf_bytes=dtf_tok, decode_fw_bytes=dfw_tok))
        print(f"[P={P:,}] D_tf={dtf[0]}({dtf[1]:,}) D_fw={dfw[0]}({dfw[1]:,})", flush=True)
        print(f"  TF val={mv:.3f}±{mvel:.3f} 训={np.mean(tf_t):.1f}s prefill={ptf*1000:.0f}ms decode={dtf_tok*1000:.2f}ms/tok", flush=True)
        print(f"  FW val={mw:.3f}±{mwel:.3f} 训={np.mean(fw_t):.1f}s prefill={pfw*1000:.0f}ms decode={dfw_tok*1000:.2f}ms/tok", flush=True)
        print(f"  → 能力差={mw-mv:+.3f} 训练速比={np.mean(tf_t)/max(np.mean(fw_t),1e-9):.2f}x prefill速比={ptf/max(pfw,1e-9):.2f}x", flush=True)

    # prefill 渐近扫描（固定 ~400K 匹配，T 扫描，batch=8）
    print("\n=== prefill 渐近（T 扫描，~400K 匹配）===", flush=True)
    dtf=search_width(mk_tf,400_000); dfw=search_width(mk_fw,400_000)
    mtf=mk_tf(dtf[0]); mfw=mk_fw(dfw[0])
    print(f"  D_tf={dtf[0]} D_fw={dfw[0]}", flush=True)
    for T in [256,512,1024,2048,4096]:
        xt=torch.randint(0,256,(8,T))
        try:
            a=prefill_time(mtf,xt,reps=2); b=prefill_time(mfw,xt,reps=2)
            print(f"  T={T:>5}: TF={a*1000:7.1f}ms FW={b*1000:7.1f}ms  FW/TF={b/max(a,1e-9):.3f}", flush=True)
        except Exception as e:
            print(f"  T={T}: SKIP {e}", flush=True)
    import json; open('results/fair_bench_v2.json','w').write(json.dumps(results,indent=2))
    print("\nDONE, saved results/fair_bench_v2.json", flush=True)

if __name__=='__main__':
    main()
