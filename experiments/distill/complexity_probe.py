"""复杂度探针: 扫序列长度 T, 测前向 wall-clock 翻倍比 (军规: log2 斜率, O(T)=1, O(T²)=2)。

对比: fusedfw(纯rho, 应O(T)) vs fused_fw_la_cycle(QR@KR.mT, 应O(T²)) vs bdh_qwen(O(T²))。
方法: 同 batch=1, 扫 T=128/256/512/1024/2048, 每档 warmup+多次取均值, torch.no_grad 测纯前向。
"""
import torch, time, json
import torch.nn.functional as F

def bench(model, T_list, runs=10, warmup=2):
    model.cuda().eval()
    res = {}
    for T in T_list:
        x = torch.randint(0, 151936, (1, T)).cuda()
        for _ in range(warmup):
            with torch.no_grad():
                lg, _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            with torch.no_grad():
                lg, _ = model(x)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / runs * 1000
        res[T] = ms
    return res

def log2_doubling(res):
    """每档翻倍的 log2(T2/T1) 斜率。"""
    Ts = sorted(res)
    ratios = []
    for i in range(1, len(Ts)):
        t_ratio = res[Ts[i]] / res[Ts[i-1]]
        seq_ratio = Ts[i] / Ts[i-1]
        ratios.append(round((t_ratio) / (seq_ratio) * 2, 2))  # 每序列翻倍时的耗时翻倍比
    return ratios

if __name__ == '__main__':
    from dynfw.models.fused_fw_qwen import FusedFWQwen
    from dynfw.models.fused_fw_la_cycle import BDHBlockCycleLM
    from dynfw.models.bdh_qwen import BDHQwen
    T_list = [128, 256, 512, 1024, 2048]

    models = {
        'fusedfw(纯rho,O(T))': FusedFWQwen(D=128, N=512, k=16, vocab=151936, use_ffn=True, n_layer=1),
        'la_cycle(steps=2)': BDHBlockCycleLM(D=128, nh=4, vocab=151936, n_layer=1, steps=2, mlp_mult=128),
        'bdh_qwen(O(T²))': BDHQwen(D=128, n_layer=1, nh=4, mlp_mult=128, vocab=151936, dropout=0.0),
    }
    out = {}
    for name, m in models.items():
        print(f"\n=== {name} ===")
        r = bench(m, T_list)
        ratios = log2_doubling(r)
        out[name] = {'ms': r, 'log2_slope(≈翻倍比)': ratios}
        for T, ms in r.items():
            print(f"  T={T:5d}: {ms:8.1f} ms")
        print(f"  → 每序列翻倍耗时翻倍比 = {ratios} (O(T)≈1.0, O(T²)≈2.0)")
        del m; torch.cuda.empty_cache()

    print("\n=== 复杂度结论 ===")
    for name, d in out.items():
        slopes = d['log2_slope(≈翻倍比)']
        print(f"  {name}: 平均斜率={sum(slopes)/len(slopes):.2f} -> {'O(T) 线性' if sum(slopes)/len(slopes)<1.3 else ('O(T²) 平方' if sum(slopes)/len(slopes)>1.6 else '介于其间/数据噪声')}")
    with open('/tmp/complexity_probe.json','w') as f:
        json.dump(out, f, indent=2)
    print("[已存] /tmp/complexity_probe.json")
