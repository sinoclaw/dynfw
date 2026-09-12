"""资源可行性验证: la_cycle(BDHBlockCycleLM) 在 RTX 4090 24GB 上的可训规模探针。

目标: 爸爸要用我们架构训个'小钢炮'打败 MiniCPM5-2B(2.52B, 42层, 128K上下文)。
先诚实测: 单卡 24GB 能训多大参数/多长上下文(显存+每step耗时), 定位临界点。

方法: 扫多档 (D, n_layer, T), bf16 混合精度, 测 forward+backward 峰值显存 + 每step 耗时。
注意: la_cycle 用 BDH 二维注意 => O(T²), 长上下文会爆显存/慢, 须实测。
"""
import torch, time, gc, json
import torch.nn.functional as F


def probe(D, n_layer, nh=16, T=512, vocab=151936, steps=2):
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    m = BDHBlockCycleLM(D=D, nh=nh, vocab=vocab, n_layer=n_layer, steps=steps,
                        mlp_mult=128).to('cuda')   # 用 float32 (RoPE freqs 需保持 float32)
    m.train()
    x = torch.randint(0, vocab, (1, T)).cuda()
    tgt = torch.randint(0, vocab, (1, T)).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    nparam = m.np()
    # 预热
    for _ in range(2):
        lg, loss = m(x, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    for _ in range(3):
        opt.zero_grad()
        lg, loss = m(x, tgt)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {'params_M': round(nparam/1e6, 1), 'peak_GB': round(peak, 2),
            'per_step_s': round(dt, 2)}


if __name__ == '__main__':
    print("==== la_cycle 可训规模探针 (4090 24GB, float32) ====")
    # 扫参数规模: D 增大 + n_layer 增大; 固定 T=512, nh=16
    configs = [
        dict(D=128, n_layer=8),   # 小
        dict(D=256, n_layer=8),   # 中
        dict(D=256, n_layer=16),
        dict(D=512, n_layer=16),
        dict(D=512, n_layer=24),
        dict(D=768, n_layer=24),  # 较大
        dict(D=1024, n_layer=32), # 接近 2B 量级?
    ]
    T = 512
    results = []
    for c in configs:
        try:
            r = probe(**c, T=T)
            label = f"D={c['D']} L={c['n_layer']} T={T}"
            r['config'] = label
            results.append(r)
            print(f"  {label:28s}  params={r['params_M']:6.1f}M  peak={r['peak_GB']:5.2f}GB  step={r['per_step_s']:6.2f}s")
        except Exception as e:
            label = f"D={c['D']} L={c['n_layer']} T={T}"
            print(f"  {label:28s}  FAILED (OOM/超时): {str(e)[:80]}")
        gc.collect(); torch.cuda.empty_cache()

    # 再测上下文: D=256,L=8, 扫 T (O(T²) 看会不会爆)
    print("\n--- 上下文长度扫描 (O(T²) 实测), D=256 L=16 ---")
    for T in [512, 1024, 2048, 4096]:
        try:
            r = probe(D=256, n_layer=16, T=T)
            print(f"  T={T:5d}  params={r['params_M']:6.1f}M  peak={r['peak_GB']:5.2f}GB  step={r['per_step_s']:6.2f}s")
        except Exception as e:
            print(f"  T={T:5d}  FAILED: {str(e)[:80]}")
        gc.collect(); torch.cuda.empty_cache()

    print("\n[参考] MiniCPM5-2B: 2.52B 总参, 1.98B 非embedding, 42层, 128K上下文")
    with open('/tmp/la_cycle_probe.json','w') as f:
        json.dump(results, f, indent=2)
