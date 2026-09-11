"""分块 KL 蒸馏损失 —— 真实模型对拍验证（loss 等价 / 参数梯度等价 / 峰值显存 / 单步耗时）

对拍口径：同一模型、同一输入、同一份教师 logits，只换 loss 实现：
  路径A(现状)  : s_lg = model.forward_logits(x); full_kl_loss(s_lg.float(), t_lg)
  路径B(方案1) : h = model.forward_hidden(x); chunked_kl_loss(h, *model.head_params(), t_lg, chunk)
"""
import sys, json, time, torch
sys.path.insert(0, '/data/dynfw')
import torch.nn.functional as F
from dynfw.models.transformer import TF_sdpa
from dynfw.models.fused_fw_full import FusedFWFull
from dynfw.training.chunked_kl import chunked_kl_loss, full_kl_loss

V, D, NH = 151936, 128, 4
B, T = 8, 1024
TEMP = 1.0
dev = 'cuda'


def build(arch):
    torch.manual_seed(0)
    if arch == 'tf':
        m = TF_sdpa(D=D, nh=NH, n_layer=2, vocab=V, maxT=T)
    elif arch == 'fusedfw_full':
        m = FusedFWFull(D=D, N=512, k=16, nh=NH, mlp_mult=128, vocab=V,
                        use_ffn=False, n_layer=1, use_softmax=False, tie=False)
    return m.to(dev)


def grads_of(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def peak_reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_now():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 30


def run_arch(arch, chunks=(256, 64)):
    print(f"\n{'='*78}\n[arch={arch}] B={B} T={T} V={V} D={D}\n{'='*78}", flush=True)
    model = build(arch)
    torch.manual_seed(1)
    x = torch.randint(0, V, (B, T), device=dev)
    # 固定教师 logits（独立随机 head，fp32，作为离线蒸馏场景的 t_lg）
    torch.manual_seed(2)
    Wt = torch.nn.Linear(D, V, bias=False, device=dev, dtype=torch.float32)
    with torch.no_grad():
        h_ref = model.forward_hidden(x).detach()
        T_LG = Wt(h_ref.float())
    print(f"教师 logits: {tuple(T_LG.shape)} {T_LG.dtype} = {T_LG.numel()*4/2**30:.2f} GiB", flush=True)

    out = {'arch': arch, 'B': B, 'T': T, 'V': V, 'D': D}
    W0, b0 = model.head_params()

    # ---- 路径 A：现状（全量）----
    model.zero_grad(set_to_none=True)
    peak_reset()
    t0 = time.time()
    s_lg = model.forward_logits(x) if hasattr(model, 'forward_logits') else model.forward(x)[0]
    lossA = full_kl_loss(s_lg.float(), T_LG, TEMP)
    lossA.backward()
    torch.cuda.synchronize()
    dtA = time.time() - t0
    peakA = peak_now()
    gA = grads_of(model)
    lossA_v = lossA.item()
    print(f"A 全量  loss={lossA_v:.6f}  peak={peakA:6.2f} GiB  {dtA:5.1f}s", flush=True)
    del s_lg, lossA
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    out['A'] = {'loss': lossA_v, 'peak_gib': round(peakA, 2), 'sec': round(dtA, 1)}

    # ---- 路径 B：分块 ----
    for chunk in chunks:
        model.zero_grad(set_to_none=True)
        peak_reset()
        t0 = time.time()
        h = model.forward_hidden(x)
        lossB = chunked_kl_loss(h, W0, b0, T_LG, chunk=chunk, temperature=TEMP)
        lossB.backward()
        torch.cuda.synchronize()
        dtB = time.time() - t0
        peakB = peak_now()
        lossB_v = lossB.item()
        # 等价性：loss + 全部参数梯度
        gB = grads_of(model)
        gl, gw, gn = 0.0, 0.0, 0
        for n in gA:
            a, b = gA[n], gB[n]
            if a is None and b is None:
                continue
            assert a is not None and b is not None, f'grad 存在性不一致: {n}'
            d = (a.float() - b.float()).abs().max().item()
            scale = a.float().abs().max().item() + 1e-12
            gl = max(gl, d / scale)
            gw = max(gw, d)
            gn += 1
        print(f"B chunk={chunk:<4} loss={lossB_v:.6f}  peak={peakB:6.2f} GiB  {dtB:5.1f}s  "
              f"|Δloss|={abs(lossB_v-lossA_v):.3e}  梯度相对maxdiff={gl:.3e} (参数量{gn})", flush=True)
        out[f'B_chunk{chunk}'] = {'loss': lossB_v, 'peak_gib': round(peakB, 2), 'sec': round(dtB, 1),
                                  'dloss_abs': abs(lossB_v - lossA_v), 'grad_rel_maxdiff': gl}
        del h, lossB
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    del model, T_LG, Wt
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    res = [run_arch('tf'), run_arch('fusedfw_full')]
    print('\n' + '=' * 78)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"GPU total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
