"""预算探测：600M 目标配置能不能跑 + 真吞吐（硬数）。
目标(方案A记忆): D1024 / N2048 / nh16 / 3层，教师 Qwen3-0.6B vocab=151936。
分别探测 fusedfw_full（我们的）+ tf（对照），报告 结构参数/词表税/显存/每步耗时/tok/s。
"""
import os, torch, time, json
import torch.nn.functional as F

def report(name, model, vocab, batch=4, block=256, steps=5):
    model = model.cuda()
    n_total = model.np()
    # 词表税 (untied): 2*vocab*D ; 取 D
    try:
        D = model.D
    except Exception:
        D = getattr(model, 'd_model', None)
    print(f"=== {name} ===")
    print(f"  total params      = {n_total/1e6:.1f} M")
    # 词表税
    if D is not None:
        vocab_tax = 2*vocab*D if not getattr(model, 'tie', False) else vocab*D
        print(f"  vocab tax(eff)    = {vocab_tax/1e6:.1f} M")
        print(f"  struct (total-tax)= {(n_total-vocab_tax)/1e6:.1f} M")
    # 前向+反向+step 计时
    x = torch.randint(0, vocab, (batch, block)).cuda()
    logits = None
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for _ in range(steps):
        if hasattr(model, 'forward'):
            out = model.forward(x)
            s_lg = out[0] if isinstance(out, tuple) else out
        else:
            s_lg = model.forward_logits(x)
        t_lg = torch.randn_like(s_lg)  # 占位 logits，只测算力
        loss = F.kl_div(F.log_softmax(s_lg, dim=-1), F.softmax(t_lg, dim=-1), reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.time() - t0
    per_step = wall/steps
    tokens = batch*block
    print(f"  per-step = {per_step*1000:.0f} ms | tokens/step = {tokens} | tok/s(训练) = {tokens/per_step:.0f}")
    print(f"  GPU mem  = {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    # 峰值显存
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.max_memory_allocated()/1e9
    model.to('cpu')
    import gc; gc.collect(); torch.cuda.empty_cache()
    return {'name': name, 'total_M': n_total/1e6, 'per_step_ms': per_step*1000,
            'tok_per_s': tokens/per_step, 'gpu_mem_GB': mem}

if __name__ == '__main__':
    from dynfw.models.fused_fw_full import FusedFWFull
    from dynfw.models.transformer import TF_sdpa
    vocab = 151936
    res = []
    # 我们的架构：方案A D1024 N2048 nh16 3层
    res.append(report('fusedfw_full_D1024_N2048_nh16_L3',
               FusedFWFull(D=1024, N=2048, k=16, nh=16, mlp_mult=128, vocab=vocab,
                           use_ffn=False, n_layer=3, use_softmax=False, tie=False), vocab))
    # TF 对照：同 D1024 nh16 3层
    res.append(report('tf_D1024_nh16_L3',
               TF_sdpa(D=1024, nh=16, n_layer=3, vocab=vocab, maxT=256), vocab))
    print("\n=== PROBE RESULT ===")
    print(json.dumps(res, indent=2))
