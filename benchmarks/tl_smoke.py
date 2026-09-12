"""TileLang 在 A800(SM80) 上的最小冒烟：真编译 + 真跑 + 真对数值。"""
import torch
import tilelang
import tilelang.language as T

print('tilelang', tilelang.__version__, '| torch', torch.__version__)
print('device', torch.cuda.get_device_name(0), '| cc', torch.cuda.get_device_capability())


@tilelang.jit(out_idx=[2])
def make(M, Nn, K, blk=64, threads=128):
    @T.prim_func
    def gemm(A: T.Tensor([M, K], T.bfloat16),
             Bm: T.Tensor([Nn, K], T.bfloat16),
             C: T.Tensor([M, Nn], T.bfloat16)):
        with T.Kernel(T.ceildiv(Nn, blk), T.ceildiv(M, blk), threads=threads) as (bx, by):
            A_s = T.alloc_shared([blk, K], T.bfloat16)
            B_s = T.alloc_shared([blk, K], T.bfloat16)
            C_f = T.alloc_fragment([blk, blk], T.float32)
            T.copy(A[by * blk, 0], A_s)
            T.copy(Bm[bx * blk, 0], B_s)
            T.clear(C_f)
            T.gemm(A_s, B_s, C_f, transpose_B=True)
            T.copy(C_f, C[by * blk, bx * blk])
    return gemm


M = Nn = K = 256
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(Nn, K, device='cuda', dtype=torch.bfloat16)
k = make(M, Nn, K)
c = k(a, b)
ref = (a.float() @ b.float().T)
md = (c.float() - ref).abs().max().item()
print(f'GEMM compiled & ran. maxdiff vs torch = {md:.3e}')
print('SMOKE_PASS' if md < 1.0 else 'SMOKE_SUSPECT')
