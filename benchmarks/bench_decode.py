"""decode 侧实测：自研架构 vs TF —— 每步延迟 / 状态内存 vs 上下文长度。

核心口径（架构固有，不依赖实现）：
  自研 decode 状态 = 窗口 k,v(≤W) + 定长 fast-weight M [B,nh,N,D]   → 与 T 无关
  TF   decode 状态 = KV cache [B,nh,T,2*hd]                        → ∝ T

实现公平性：两边都用【预分配缓存】（slicing 是 view，无 cat 的 O(T) 拷贝），都 eager。
⚠️ 诚实标注：自研解码是手写第一版，TF 走 PyTorch 原生 SDPA 路径 —— 实现成熟度不对等。

用法: PYTHONPATH=/data/dynfw python benchmarks/bench_decode.py
"""
import sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/dynfw')
from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
from dynfw.models.fused_fw_fw_cycle_opt import FWAttentionOpt
from dynfw.models.transformer import TF_sdpa

VOCAB, D, NH, NL, W = 50257, 256, 8, 6, 256
BIG_T = 262144


# ===========================================================================
# 自研增量解码
# ===========================================================================
class DynDecoder:
    def __init__(self, lm, B=1, dt=torch.bfloat16, dev='cuda', amp=True):
        self.lm = lm; self.B = B; self.W = lm.W; self.pos = 0; self.amp = amp
        self.states = []
        for blk in lm.blocks:
            D = blk.D; nh = blk.config.n_head
            N = blk.config.mlp_internal_dim_multiplier * D // nh
            self.states.append(dict(
                # 训练里 new_mem 起点是 fp32 且用 .float() 累积 → 解码侧也必须 fp32 存，
                # 否则单是 dtype 差就会让 decode-vs-forward 的 logits 差放大数倍
                mem=torch.zeros(B, nh, N, D, device=dev, dtype=torch.float32),
                kw=torch.zeros(B, nh, self.W, N, device=dev, dtype=dt),
                vw=torch.zeros(B, 1, self.W, D, device=dev, dtype=dt),
                n=0))

    def state_bytes(self):
        tot = 0
        for s in self.states:
            tot += s['mem'].numel() * s['mem'].element_size()
            tot += s['kw'].numel() * s['kw'].element_size()
            tot += s['vw'].numel() * s['vw'].element_size()
        return tot

    def _blk(self, blk, st, x):
      with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
        C = blk.config; D = blk.D; nh = C.n_head
        N = C.mlp_internal_dim_multiplier * D // nh
        B = x.shape[0]; pos = self.pos; W = self.W
        x = blk.ln(x)
        mem = st['mem']
        for _ in range(blk.steps):
            x_latent = x @ blk.encoder                       # [B,nh,1,N]
            x_sparse = F.relu(x_latent)
            rr = torch.full((1, 1, 1, 1), float(pos), device=x.device, dtype=torch.float32)
            QR = FWAttentionOpt.rope_fast(rr * blk.attn.freqs, x_sparse)   # [B,nh,1,N]
            # 进入新块：先把上一块结算进定长 M
            if pos % W == 0 and st['n'] > 0:
                n = st['n']
                mem = mem + torch.einsum('bhmd,bme->bhde',
                                         st['kw'][:, :, :n].float(), st['vw'][:, 0, :n].float())
                st['n'] = 0
            n = st['n']
            st['kw'][:, :, n] = QR[:, :, 0]
            st['vw'][:, 0, n] = x[:, 0, 0]
            st['n'] = n + 1
            # 块内因果注意（只用当前块内已见 token，≤W）
            sim = torch.einsum('bhd,bhmd->bhm', QR[:, :, 0], st['kw'][:, :, :n + 1])
            p = torch.softmax(sim.float(), dim=-1)
            agg = torch.einsum('bhm,bme->bhe', p, st['vw'][:, 0, :n + 1])
            # 跨块检索（定长 M）
            retr = torch.einsum('bhd,bhde->bhe', QR[:, :, 0], mem)
            yKV = blk.ln((agg + retr).unsqueeze(2))            # [B,nh,1,D]
            y_latent = yKV @ blk.encoder_v
            y_sparse = F.relu(y_latent)
            xy = x_sparse * y_sparse
            yMLP = xy.transpose(1, 2).reshape(B, 1, 1, N * nh) @ blk.decoder
            x = blk.ln(x + blk.ln(yMLP))
        st['mem'] = mem
        return x

    @torch.no_grad()
    def prefill(self, seq, W_mult=True):
        """整段前向一次性建状态（真实服务做法）。要求 len(seq) % W == 0，
        这样每个块都是完整的 → block 返回的 memories 正好是 pos=T 该有的 M、窗口为空。"""
        lm = self.lm; T = seq.shape[1]
        assert (not W_mult) or T % self.W == 0, f'prefill 长度须为 W={self.W} 的倍数'
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            h = lm.ln(lm.e(seq).unsqueeze(1))                  # [B,1,T,D]
            for bi, blk in enumerate(lm.blocks):
                h, mem = blk(h, None)
                self.states[bi]['mem'] = mem.float()
                self.states[bi]['n'] = 0
        self.pos = T

    @torch.no_grad()
    def step(self, tok):
      with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
        lm = self.lm; B = tok.shape[0]
        # ⚠️ 必须补出时间维 → [B,1,1,D]；只 unsqueeze(1) 得 [B,1,D] 会让 block 内的
        #    `QR[:,:,0]` 变成 [B,nh] 被【静默广播】填进 [B,nh,N] 槽位（不报错、结果全错）
        h = lm.ln(lm.e(tok).unsqueeze(1).unsqueeze(1))         # [B,1,1,D]
        for bi, blk in enumerate(lm.blocks):
            h = self._blk(blk, self.states[bi], h)
        self.pos += 1
        return lm.head(h.view(B, 1, lm.D))


# ===========================================================================
# TF 增量解码（预分配 KV cache）
# ===========================================================================
class TFDecoder:
    def __init__(self, m, B=1, maxlen=4096, dt=torch.bfloat16, dev='cuda'):
        self.m = m; self.B = B; self.maxlen = maxlen; self.pos = 0
        D = m.D; h = m.nh; hd = D // h
        self.kbuf = [torch.zeros(B, h, maxlen, hd, device=dev, dtype=dt) for _ in m.blocks]
        self.vbuf = [torch.zeros(B, h, maxlen, hd, device=dev, dtype=dt) for _ in m.blocks]

    def state_bytes(self):
        n = sum(t.numel() for t in self.kbuf) + sum(t.numel() for t in self.vbuf)
        return n * self.kbuf[0].element_size()

    @torch.no_grad()
    def prefill(self, seq):
        m = self.m; B, T = seq.shape; D = m.D; h = m.nh; hd = D // h
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = m.e(seq) + m.pos[:T].unsqueeze(0)              # [B,T,D]
            for bi, (ln1, qkv, proj, ln2, w1, act, w2) in enumerate(m.blocks):
                xx = ln1(y); q, k, v = qkv(xx).chunk(3, dim=-1)
                q = q.view(B, T, h, hd).transpose(1, 2)
                k = k.view(B, T, h, hd).transpose(1, 2)
                v = v.view(B, T, h, hd).transpose(1, 2)
                self.kbuf[bi][:, :, :T] = k
                self.vbuf[bi][:, :, :T] = v
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = o.transpose(1, 2).reshape(B, T, D)
                y = y + proj(o); y = y + w2(act(w1(ln2(y))))
        self.pos = T

    @torch.no_grad()
    def step(self, tok):
        with torch.autocast('cuda', dtype=torch.bfloat16):   # ⚠️ 必须包：否则 q 是 fp32、缓存是 bf16
            m = self.m; B = tok.shape[0]; D = m.D; h = m.nh; hd = D // h; p = self.pos
            y = (m.e(tok) + m.pos[p]).unsqueeze(1)              # [B,1,D]
            for bi, (ln1, qkv, proj, ln2, w1, act, w2) in enumerate(m.blocks):
                xx = ln1(y)
                q, k, v = qkv(xx).chunk(3, dim=-1)
                q = q.view(B, 1, h, hd).transpose(1, 2)
                k = k.view(B, 1, h, hd).transpose(1, 2)
                v = v.view(B, 1, h, hd).transpose(1, 2)
                self.kbuf[bi][:, :, p:p + 1] = k
                self.vbuf[bi][:, :, p:p + 1] = v
                o = F.scaled_dot_product_attention(q, self.kbuf[bi][:, :, :p + 1],
                                                   self.vbuf[bi][:, :, :p + 1], is_causal=False)
                o = o.transpose(1, 2).reshape(B, 1, D)
                y = y + proj(o)
                y = y + w2(act(w1(ln2(y))))
            self.pos += 1
            return m.h(m.ln(y))


# ===========================================================================
# 静态形状版解码：窗口恒为 W + 有效掩码；消掉所有 Python 控制流
#   - 折叠只在块边界发生 → 用标量 isb(0/1) 乘代替 if
#   - 槽位写入用 one-hot where 代替动态索引（避免每个 slot 触发一次重编译）
#   - token 从预分配 buffer 读 → 地址固定 → CUDA Graph 可用
# 语义等价性：掩码槽位必须填 -inf（不能填 0），否则 softmax 分母变化 → 不是同一函数
# ===========================================================================
class DynDecoderS:
    def __init__(self, lm, B=1, dev='cuda', dt=torch.bfloat16, W=None, amp=True):
        self.lm = lm; self.B = B; self.W = W or lm.W; self.pos = 0
        self.dt = dt; self.dev = dev; self.amp = amp
        self.tokbuf = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.slot = torch.zeros((), dtype=torch.long, device=dev)
        self.isb = torch.zeros((), dtype=torch.float32, device=dev)
        # ⚠️ rope 的位置必须是【输入张量】而不是 Python 常量 —— 否则会被烘进 CUDA Graph，
        #    replay 时永远用同一个位置（参数一样、成本一样，但算的是错的东西）
        self.posbuf = torch.zeros((), dtype=torch.long, device=dev)
        # 输出写进固定缓冲：CUDA Graph 的产物必须能从固定张量读回（图里不能新建可见输出）
        self.outbuf = torch.zeros(B, 1, lm.vocab, device=dev, dtype=dt)
        self.ar = torch.arange(self.W, device=dev)
        st = []
        for blk in lm.blocks:
            D = blk.D; nh = blk.config.n_head
            N = blk.config.mlp_internal_dim_multiplier * D // nh
            st.append(dict(mem=torch.zeros(B, nh, N, D, device=dev, dtype=torch.float32),
                           kw=torch.zeros(B, nh, self.W, N, device=dev, dtype=dt),
                           vw=torch.zeros(B, 1, self.W, D, device=dev, dtype=dt)))
        self.states = st

    def state_bytes(self):
        t = 0
        for s in self.states:
            for v in s.values():
                t += v.numel() * v.element_size()
        return t

    def _blk(self, blk, s, x):
        C = blk.config; D = blk.D; nh = C.n_head
        N = C.mlp_internal_dim_multiplier * D // nh
        B = x.shape[0]; W = self.W
        x = blk.ln(x)
        for _ in range(blk.steps):
            x_latent = x @ blk.encoder
            x_sparse = F.relu(x_latent)
            r = self.posbuf.float().view(1, 1, 1, 1)
            QR = FWAttentionOpt.rope_fast(r * blk.attn.freqs, x_sparse)     # [B,nh,1,N]
            # ① 块边界：把上一块的完整窗口折叠进定长 M（isb 非边界时为 0，等价于跳过）
            fold = torch.einsum('bhmd,bme->bhde', s['kw'].float(), s['vw'][:, 0].float())
            # ⚠️⚠️ 必须【原地写】(add_/copy_)。若写成 `s['mem'] = s['mem'] + ...` 这种「重新绑定到新张量」，
            #    CUDA Graph 捕获的是「旧张量地址」作输入、结果写到新地址 → replay 时永远读同一个旧地址，
            #    状态不会跨步累积（实测 maxdiff 0.32、top-1 崩）。闸门C 就是为抓这个而设。
            s['mem'].add_(self.isb * fold)
            # ② 写入当前槽位（one-hot where，无动态索引）
            oh = (self.ar == self.slot).view(1, 1, W, 1)
            # QR 已是 [B,nh,1,N]、x 已是 [B,1,1,D] → 直接与 oh 广播，勿再 unsqueeze
            s['kw'].copy_(torch.where(oh, QR, s['kw']))
            s['vw'].copy_(torch.where(oh, x, s['vw']))
            # ③ 块内因果注意（掩码槽位 -inf）
            sim = torch.einsum('bhd,bhmd->bhm', QR[:, :, 0], s['kw'])
            keep = (self.ar <= self.slot).view(1, 1, W)
            sim = sim.masked_fill(~keep, float('-inf'))
            p = torch.softmax(sim.float(), dim=-1)
            agg = torch.einsum('bhm,bme->bhe', p, s['vw'][:, 0])
            retr = torch.einsum('bhd,bhde->bhe', QR[:, :, 0], s['mem'])
            yKV = blk.ln((agg + retr).unsqueeze(2))
            y_latent = yKV @ blk.encoder_v
            y_sparse = F.relu(y_latent)
            xy = x_sparse * y_sparse
            yMLP = xy.transpose(1, 2).reshape(B, 1, 1, N * nh) @ blk.decoder
            x = blk.ln(x + blk.ln(yMLP))
        return x

    @torch.no_grad()
    def step(self):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            lm = self.lm; B = self.B
            tok = self.tokbuf[:, 0]
            h = lm.ln(lm.e(tok).unsqueeze(1).unsqueeze(1))
            for i, blk in enumerate(lm.blocks):
                h = self._blk(blk, self.states[i], h)
            lg = lm.head(h.view(B, 1, lm.D))
            self.outbuf.copy_(lg)
        return self.outbuf

    def advance(self):
        """状态推进必须【在图外】调用 —— 这样 step() 才能整体被 CUDA Graph 捕获。"""
        self.pos += 1
        self.posbuf.fill_(self.pos)
        m = self.pos % self.W
        self.slot.fill_(m)
        self.isb.fill_(1.0 if (m == 0 and self.pos > 0) else 0.0)

    @torch.no_grad()
    def prefill(self, seq):
        lm = self.lm; T = seq.shape[1]
        assert T % self.W == 0
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            h = lm.ln(lm.e(seq).unsqueeze(1))
            for i, blk in enumerate(lm.blocks):
                h, mem = blk(h, None)
                # 原地拷贝：捕获后再调 prefill 做状态重置时，不能重新绑定张量地址
                self.states[i]['mem'].copy_(mem.float())
                self.states[i]['kw'].zero_(); self.states[i]['vw'].zero_()
        self.pos = T
        self.posbuf.fill_(T); self.slot.fill_(0); self.isb.fill_(0.0)


class TFDecoderS:
    """TF 的「固定形状」版：maxlen 定长 KV 缓冲 + bool 掩码 → 也能被 CUDA Graph 捕获。

    目的：量化「TF 为了拿到固定形状要付什么代价」——这是自研「定长状态」优势的对照面。
    ⚠️ 传 attn_mask 会让 SDPA 失去 FlashAttention（回退 mem_efficient/math），代价必须如实报。
    """
    def __init__(self, m, B=1, maxlen=4096, dev='cuda', dt=torch.bfloat16):
        self.m = m; self.B = B; self.maxlen = maxlen
        D = m.D; h = m.nh; hd = D // h
        self.kbuf = torch.zeros(B, h, maxlen, hd, device=dev, dtype=dt)
        self.vbuf = torch.zeros(B, h, maxlen, hd, device=dev, dtype=dt)
        self.tokbuf = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.posbuf = torch.zeros((), dtype=torch.long, device=dev)
        self.ar = torch.arange(maxlen, device=dev)
        self.outbuf = torch.zeros(B, 1, m.vocab, device=dev, dtype=dt)

    def state_bytes(self):
        return (self.kbuf.numel() + self.vbuf.numel()) * self.kbuf.element_size()

    @torch.no_grad()
    def step(self):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m = self.m; B = self.B; D = m.D; h = m.nh; hd = D // h
            pos = self.posbuf
            y = (m.e(self.tokbuf[:, 0]) + m.pos[pos]).unsqueeze(1)      # [B,1,D]
            mask = (self.ar <= pos).view(1, 1, 1, self.maxlen)
            for bi, (ln1, qkv, proj, ln2, w1, act, w2) in enumerate(m.blocks):
                xx = ln1(y)
                q, k, v = qkv(xx).chunk(3, dim=-1)
                q = q.view(B, 1, h, hd).transpose(1, 2)
                k = k.view(B, 1, h, hd).transpose(1, 2)
                v = v.view(B, 1, h, hd).transpose(1, 2)
                # 原地写入固定槽位
                oh = (self.ar == pos).view(1, 1, self.maxlen, 1)
                self.kbuf.copy_(torch.where(oh, k, self.kbuf))
                self.vbuf.copy_(torch.where(oh, v, self.vbuf))
                o = F.scaled_dot_product_attention(q, self.kbuf, self.vbuf, attn_mask=mask)
                o = o.transpose(1, 2).reshape(B, 1, D)
                y = y + proj(o); y = y + w2(act(w1(ln2(y))))
            self.outbuf.copy_(m.h(m.ln(y)))
        return self.outbuf

    def advance(self):
        self.posbuf.fill_(int(self.posbuf.item()) + 1)

    @torch.no_grad()
    def prefill(self, seq):
        m = self.m; B, T = seq.shape; D = m.D; h = m.nh; hd = D // h
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = m.e(seq) + m.pos[:T].unsqueeze(0)
            for bi, (ln1, qkv, proj, ln2, w1, act, w2) in enumerate(m.blocks):
                xx = ln1(y); q, k, v = qkv(xx).chunk(3, dim=-1)
                q = q.view(B, T, h, hd).transpose(1, 2)
                k = k.view(B, T, h, hd).transpose(1, 2)
                v = v.view(B, T, h, hd).transpose(1, 2)
                self.kbuf[:, :, :T].copy_(k); self.vbuf[:, :, :T].copy_(v)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = o.transpose(1, 2).reshape(B, T, D)
                y = y + proj(o); y = y + w2(act(w1(ln2(y))))
        self.posbuf.fill_(T)


def bench3(B=1, T=8192, reps=5):
    print()
    print('=' * 92)
    print(f'Part 4 公平桌：TF 固定形状化的代价（batch={B}, T={T}）')
    print('=' * 92)
    torch.manual_seed(0)
    seq = torch.randint(0, VOCAB, (B, T + 32), device='cuda')
    res = {}

    def timed(is_graph, dec):
        with torch.no_grad():
            if is_graph:
                dec.tokbuf[:, 0] = seq[:, T]
                for _ in range(3):
                    dec.step(); dec.advance()
                cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cg):
                    dec.step()
                dec.prefill(seq[:, :T])
                for k in range(2):
                    dec.tokbuf[:, 0] = seq[:, T + k]; cg.replay(); dec.advance()
                torch.cuda.synchronize(); ts = []
                for k in range(reps):
                    dec.tokbuf[:, 0] = seq[:, T + 3 + k]
                    t0 = time.time(); cg.replay(); dec.advance(); torch.cuda.synchronize()
                    ts.append(1000 * (time.time() - t0))
            else:
                for k in range(2):
                    dec.tokbuf[:, 0] = seq[:, T + k]; dec.step(); dec.advance()
                torch.cuda.synchronize(); ts = []
                for k in range(reps):
                    dec.tokbuf[:, 0] = seq[:, T + 3 + k]
                    t0 = time.time(); dec.step(); dec.advance(); torch.cuda.synchronize()
                    ts.append(1000 * (time.time() - t0))
        return sorted(ts)[len(ts) // 2]

    for name, mk in (('TF 定长+掩码', lambda: TFDecoderS(build_tf(T + 64), B=B, maxlen=T + 32)),
                     ('自研 定长', lambda: DynDecoderS(build(), B=B))):
        for graph in (False, True):
            m = mk()
            tag = name + ('+graph' if graph else '')
            try:
                m.prefill(seq[:, :T])
                med = timed(graph, m)
                res[tag] = med
                print(f'  {tag:<18} {med:7.2f} ms/step   状态 {m.state_bytes()/1e6:8.1f} MB')
            except Exception as e:
                print(f'  {tag:<18} ERR {type(e).__name__}: {str(e)[:50]}')
            del m
            try:
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            except Exception:
                pass

    tfd = TFDecoder(build_tf(T + 64), B=B, maxlen=T + 32)
    tfd.prefill(seq[:, :T])
    with torch.no_grad():
        for k in range(2):
            tfd.step(seq[:, T + k])
        torch.cuda.synchronize(); ts = []
        for k in range(reps):
            t0 = time.time(); tfd.step(seq[:, T + 3 + k]); torch.cuda.synchronize()
            ts.append(1000 * (time.time() - t0))
    res['TF 原生动态'] = sorted(ts)[len(ts) // 2]
    print(f'  {"TF 原生动态":<18} {res["TF 原生动态"]:7.2f} ms/step   状态 {tfd.state_bytes()/1e6:8.1f} MB')
    del tfd; torch.cuda.empty_cache()

    print()
    n0 = res['TF 原生动态']
    n1 = res.get('TF 定长+掩码', float('nan'))
    for k in ('TF 定长+掩码', 'TF 定长+掩码+graph', '自研 定长', '自研 定长+graph'):
        if k in res:
            print(f'    {k:<22} = {res[k]:6.2f} ms   '
                  f'(vs TF原生 {n0/res[k]:5.2f}x, vs TF定长 {n1/res[k]:5.2f}x)')


# ===========================================================================
def build(seed=0):
    torch.manual_seed(seed)
    return BDHBlockFWCycleLM(D=D, nh=NH, vocab=VOCAB, n_layer=NL,
                             steps=1, mlp_mult=4, W=W).cuda().eval()


def build_tf(maxT=BIG_T):
    return TF_sdpa(D=D, nh=NH, vocab=VOCAB, n_layer=NL, maxT=maxT).cuda().eval()


def _run_decode(lm, seq, T, **kw):
    dec = DynDecoder(lm, B=1, **kw)
    out = []
    for t in range(T):
        out.append(dec.step(seq[:, t:t + 1].squeeze(0))[:, -1, :].float())
    return torch.cat(out, 0), dec


def verify():
    print('=' * 92)
    print('Part 0 正确性闸门：增量解码 vs 整段 forward')
    print('=' * 92)
    T = 600
    torch.manual_seed(3)
    seq = torch.randint(0, VOCAB, (1, T), device='cuda')

    # 闸门 A（严格）：两边都 fp32、都关 autocast → 只该剩归约顺序噪声
    lm32 = build()
    with torch.no_grad():
        with torch.autocast('cuda', enabled=False):
            ref32 = lm32(seq, None)[0][0].float()
            got32, _ = _run_decode(lm32, seq, T, amp=False, dt=torch.float32)
    md = (ref32 - got32).abs().max().item(); sc = ref32.abs().max().item()
    print(f'  闸门A fp32(两边同路径) : maxdiff={md:.4e} 相对={md/sc:.3e} '
          f'(幅值{sc:.2f})  {"PASS" if md/sc < 1e-5 else "FAIL"}  <- 过不了就是实现写错')

    # 闸门 B（部署口径）：两边都 bf16 autocast
    lm16 = build()
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ref16 = lm16(seq, None)[0][0].float()
            got16, _ = _run_decode(lm16, seq, T, amp=True)
    md2 = (ref16 - got16).abs().max().item(); sc2 = ref16.abs().max().item()
    top1 = (ref16.argmax(-1) == got16.argmax(-1)).float().mean().item()
    print(f'  闸门B bf16(部署口径)   : maxdiff={md2:.4e} 相对={md2/sc2:.3e} (幅值{sc2:.2f})  '
          f'top-1 一致率={top1*100:.2f}%')

    # 对照 TF（PyTorch 原生路径的地板）
    tfm = build_tf(1024)
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rtf = tfm(seq, None)[0][0].float()
        tfd = TFDecoder(tfm, B=1, maxlen=T)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gtf = torch.cat([tfd.step(seq[:, t:t + 1].squeeze(0))[:, -1, :].float()
                             for t in range(T)], 0)
    mt = (rtf - gtf).abs().max().item(); st = rtf.abs().max().item()
    t1 = (rtf.argmax(-1) == gtf.argmax(-1)).float().mean().item()
    print(f'  对照 TF  bf16          : maxdiff={mt:.4e} 相对={mt/st:.3e} '
          f'(幅值{st:.2f})  top-1={t1*100:.2f}%  <- 这是原生路径地板')


def bench(B=1, marks=(512, 1024, 2048, 4096, 8192, 16384), reps=5):
    print()
    print('=' * 92)
    print(f'Part 1 每步 decode 延迟 vs 上下文长度（batch={B}，prefill 后定点测，中位×{reps}）')
    print('=' * 92)
    res = {}
    for T in marks:
        torch.manual_seed(0)
        seq = torch.randint(0, VOCAB, (B, T + 16), device='cuda')
        row = {'T': T}
        for name in ('自研', 'TF'):
            if name == '自研':
                dec = DynDecoder(build(), B=B)
                dec.prefill(seq[:, :T])
            else:
                dec = TFDecoder(build_tf(max(T + 64, 4096)), B=B, maxlen=T + 16)
                dec.prefill(seq[:, :T])
            with torch.no_grad():
                for p in (T, T + 1):
                    dec.step(seq[:, p])
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    t0 = time.time(); dec.step(seq[:, T + 2]); torch.cuda.synchronize()
                    ts.append(1000 * (time.time() - t0))
            ts.sort()
            row[name] = ts[len(ts) // 2]
            row[name + '_mb'] = dec.state_bytes() / 1e6
            del dec; torch.cuda.empty_cache()
        res[T] = row
        print(f"  T={T:<7} 自研 {row['自研']:7.2f} ms/step   TF {row['TF']:7.2f} ms/step   "
              f"比值 自研/TF = {row['自研']/row['TF']:5.2f}x   状态 "
              f"{row['自研_mb']:.2f} vs {row['TF_mb']:.2f} MB")

    print()
    print('  每步「权重 + 状态」字节账（权重两边都要读；状态是我们的定长 vs TF 的 ∝T）')
    wp = sum(p.numel() for p in build().parameters()) * 2 / 1e6
    print(f'  {"T":>7}{"我们 MB/步":>13}{"TF MB/步":>11}{"总字节比":>11}{"测到 ms 比":>12}')
    for T in marks:
        r = res[T]
        ours = wp + r['自研_mb']
        tf = wp + r['TF_mb']
        print(f'  {T:>7}{ours:>13.1f}{tf:>11.1f}{tf/ours:>10.2f}x{r["自研"]/r["TF"]:>11.2f}x')


def bench2(B=1, Ts=(2048, 8192), reps=5):
    """B′：实现成熟度对齐。自研静态版 vs 自研动态版 vs TF（动态 slicing，PyTorch 原生最佳简单实现）。"""
    print()
    print('=' * 92)
    print(f'Part 3 实现成熟度对齐（batch={B}）')
    print('=' * 92)
    print(f"  {'T':>7}{'TF(原生动态)':>15}{'自研(动态)':>14}{'自研(静态)':>14}"
          f"{'自研(静态+graph)':>18}")
    for T in Ts:
        torch.manual_seed(0)
        seq = torch.randint(0, VOCAB, (B, T + 32), device='cuda')
        cells = []

        # TF 原生（动态 slicing）
        tfd = TFDecoder(build_tf(max(T + 64, 4096)), B=B, maxlen=T + 32); tfd.prefill(seq[:, :T])
        with torch.no_grad():
            for p in (T, T + 1): tfd.step(seq[:, p])
            torch.cuda.synchronize(); ts = []
            for _ in range(reps):
                t0 = time.time(); tfd.step(seq[:, T + 2]); torch.cuda.synchronize()
                ts.append(1000 * (time.time() - t0))
        cells.append(sorted(ts)[len(ts) // 2]); del tfd; torch.cuda.empty_cache()

        # 自研（动态窗口原版）
        dm = DynDecoder(build(), B=B); dm.prefill(seq[:, :T])
        with torch.no_grad():
            for p in (T, T + 1): dm.step(seq[:, p])
            torch.cuda.synchronize(); ts = []
            for _ in range(reps):
                t0 = time.time(); dm.step(seq[:, T + 2]); torch.cuda.synchronize()
                ts.append(1000 * (time.time() - t0))
        cells.append(sorted(ts)[len(ts) // 2]); del dm; torch.cuda.empty_cache()

        # 自研（静态版 eager）
        ds = DynDecoderS(build(), B=B); ds.prefill(seq[:, :T])
        with torch.no_grad():
            for k in range(3):
                ds.tokbuf[:, 0] = seq[:, T + k]; ds.step(); ds.advance()
            torch.cuda.synchronize(); ts = []
            for r_ in range(reps):
                ds.tokbuf[:, 0] = seq[:, T + 3 + r_]
                t0 = time.time(); ds.step(); ds.advance(); torch.cuda.synchronize()
                ts.append(1000 * (time.time() - t0))
        cells.append(sorted(ts)[len(ts) // 2])
        ds_bytes = ds.state_bytes() / 1e6

        # 自研（静态版 + CUDA Graph）
        ds2 = DynDecoderS(build(), B=B); ds2.prefill(seq[:, :T])
        try:
            cg = torch.cuda.CUDAGraph()
            ds2.tokbuf[:, 0] = seq[:, T]
            for k in range(3):
                ds2.step(); ds2.advance()
            torch.cuda.synchronize()
            with torch.cuda.graph(cg):
                ds2.step()
            ds2.prefill(seq[:, :T])      # 捕获后重置状态到 pos=T（原地，不打断图绑定）
            with torch.no_grad():
                for k in range(3):
                    ds2.tokbuf[:, 0] = seq[:, T + k]; cg.replay(); ds2.advance()
                torch.cuda.synchronize(); ts = []
                for k in range(reps):
                    ds2.tokbuf[:, 0] = seq[:, T + 3 + k]
                    t0 = time.time(); cg.replay(); ds2.advance(); torch.cuda.synchronize()
                    ts.append(1000 * (time.time() - t0))
            cells.append(sorted(ts)[len(ts) // 2])
        except Exception as e:
            cells.append(float('nan'))
            print(f'    (CUDAGraph 失败: {type(e).__name__} {str(e)[:60]})')
        print(f'  {T:>7}' + ''.join(
            (f'{c:>15.2f}' if i == 0 else f'{c:>14.2f}' if i < 3 else f'{c:>18.2f}')
            for i, c in enumerate(cells)) + f'   (自研静态状态 {ds_bytes:.1f}MB)')
        del ds, ds2; torch.cuda.empty_cache()


def analytic():
    print()
    print('=' * 92)
    print('Part 2 解析账（架构固有，与实现无关）：decode 每步状态字节 / 权重字节')
    print('=' * 92)
    hd = D // NH
    print(f'  配置: D={D} nh={NH} hd={hd} N={4*D//NH} W={W} n_layer={NL} vocab={VOCAB}')
    mb = 1e6
    print(f'  我们每步状态 = W*(nh*N + D + nh*N)      = {W*(NH*128 + D + NH*128)*2/mb:.2f} MB (与 T 无关)')
    for T in (1024, 4096, 16384, 65536, 262144):
        kv = 2 * NH * T * hd * 2 / mb
        print(f'    T={T:<8} TF KV cache = 2*nh*T*hd*2B = {kv:9.2f} MB   '
              f'比值 = {kv/(W*(NH*128 + D + NH*128)*2/mb):8.1f}x')
    ours_w = sum(p.numel() for p in build().parameters()) * 2 / mb
    tf_w = sum(p.numel() for p in build_tf().parameters()) * 2 / mb
    print(f'  权重（两边都必须每步读）: 自研 {ours_w:.1f} MB / TF {tf_w:.1f} MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--marks', default='512,1024,2048,4096,8192,16384,32768')
    a = ap.parse_args()
    if a.part in ('all', '0'):
        verify()
    if a.part in ('all', '2'):
        analytic()
    if a.part in ('all', '1'):
        bench(B=a.batch, marks=[int(x) for x in a.marks.split(',')])
    if a.part in ('all', '3'):
        bench2(B=a.batch)
    if a.part in ('all', '4'):
        bench3(B=a.batch)
