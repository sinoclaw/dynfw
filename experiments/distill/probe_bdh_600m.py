"""BDH 两形态 600M 级预算探测 + 真吞吐（硬数）。
1) bdh_qwen.py  —— 语言建模形态的 BDH（dynfw 里对标 Qwen 的那条线）
2) bdh_cq 的模型 — BDH-CQ 复现（若可当 LM 用的话也在同构标注）
目标：把结构参数拉到 ~600M 量级，measure 显存/每步耗时/tok/s。
"""
import os, torch, time, json
import torch.nn.functional as F

def probe(name, model, vocab, batch=4, block=256, steps=5):
    model = model.cuda()
    n_total = model.np()
    D = getattr(model, 'D', None) or getattr(model, 'd_model', None)
    print(f"=== {name} ===")
    print(f"  total params = {n_total/1e6:.1f} M")
    if D:
        vt = 2*vocab*D if not getattr(model, 'tie', False) else vocab*D
        print(f"  vocab tax    = {vt/1e6:.1f} M | struct = {(n_total-vt)/1e6:.1f} M")
    x = torch.randint(0, vocab, (batch, block)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for _ in range(steps):
        try:
            out = model.forward(x)
            s_lg = out[0] if isinstance(out, tuple) else out
        except AttributeError:
            s_lg = model.forward_logits(x)
        t_lg = torch.randn_like(s_lg)
        loss = F.kl_div(F.log_softmax(s_lg, dim=-1), F.softmax(t_lg, dim=-1), reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.time() - t0
    per = wall/steps
    tok = batch*block
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.max_memory_allocated()/1e9
    print(f"  per-step={per*1000:.0f}ms | tok/s(train)={tok/per:.0f} | peak mem={mem:.1f}GB")
    model.to('cpu')
    import gc; gc.collect(); torch.cuda.empty_cache()
    return {'name': name, 'total_M': n_total/1e6, 'per_step_ms': per*1000,
            'tok_s': tok/per, 'gpu_mem_GB': mem}

if __name__ == '__main__':
    from dynfw.models.bdh_qwen import BDHQwen
    vocab = 151936
    res = []
    # 语言建模形态 BDH：逐步放大，找 ~600M 结构参数
    for name, kwargs in [
        ('bdh_qwen_D512_nh8_L4_mlp128', dict(D=512, nh=8, n_layer=4, mlp_mult=128, vocab=vocab, dropout=0.0)),
        ('bdh_qwen_D768_nh8_L8_mlp128', dict(D=768, nh=8, n_layer=8, mlp_mult=128, vocab=vocab, dropout=0.0)),
        ('bdh_qwen_D768_nh8_L16_mlp64', dict(D=768, nh=8, n_layer=16, mlp_mult=64, vocab=vocab, dropout=0.0)),
    ]:
        try:
            res.append(probe(name, BDHQwen(**kwargs), vocab))
        except Exception as e:
            print(f"{name} FAILED: {e}")
    print("\n=== BDH PROBE RESULT ===")
    print(json.dumps(res, indent=2))
