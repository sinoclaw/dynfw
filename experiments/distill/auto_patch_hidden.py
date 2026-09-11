"""v3：从 .bak 恢复后重做注入；插入锚点限制在【目标 forward 所在类】内（v2 会插进前面的同类名辅助类）"""
import re, sys, pathlib

APPLY = '--apply' in sys.argv
ROOT = pathlib.Path('/data/dynfw/dynfw/models')
FILES = ['bdh_gla.py', 'bdh_gla_v2.py', 'bdh_gla_v3.py', 'bdh_qwen.py', 'bdh_rawfw_qwen.py',
         'fused_fw_cycle.py', 'fused_fw_dla_cycle.py', 'fused_fw_dla_topk_cycle.py',
         'fused_fw_fw_cycle.py', 'fused_fw_gdn_cycle.py', 'fused_fw_la.py', 'fused_fw_la_cycle.py',
         'fused_fw_lin.py', 'fused_fw_qwen.py', 'fused_fw_rawfw_cycle.py', 'fused_fw_rec.py']
SPECIAL = ['fused_fw_full_shared.py']


def find_target_forward(lines):
    for i, ln in enumerate(lines):
        m = re.match(r'(\s*)def forward\(self,\s*(\w+)', ln)
        if not m:
            continue
        ind = len(m.group(1))
        body, j = [], i + 1
        while j < len(lines):
            l = lines[j]
            if l.strip() == '':
                body.append(j); j += 1; continue
            if len(l) - len(l.lstrip()) <= ind:
                break
            body.append(j); j += 1
        for idx in body:
            mm = re.match(r'(\s*)(?:lg|logits)\s*=\s*(.+)$', lines[idx])
            if mm and ('self.head' in mm.group(2) or 'self.lm_head' in mm.group(2)):
                return i, m.group(2), body, idx, mm.group(2), len(mm.group(1))
    return None


def class_bounds(lines, dstart):
    cs = 0
    for i in range(dstart, -1, -1):
        if re.match(r'class ', lines[i]):
            cs = i
            break
    ce = len(lines)
    for i in range(cs + 1, len(lines)):
        if re.match(r'class ', lines[i]):
            ce = i
            break
    return cs, ce


def hidden_expr(expr):
    if re.match(r'self\.(head|h|lm_head)\(', expr):
        k = expr.index('('); depth = 0
        for p in range(k, len(expr)):
            if expr[p] == '(':
                depth += 1
            elif expr[p] == ')':
                depth -= 1
                if depth == 0:
                    return expr[k + 1:p]
        raise ValueError('unbalanced: ' + expr)
    if ' @ ' in expr:
        return expr.split(' @ ')[0].strip()
    raise ValueError('cannot infer hidden: ' + expr)


def head_ret(expr, src):
    if 'self.lm_head' in expr and 'self.head' not in expr:
        assert 'self.lm_head = nn.Parameter(' in src, 'lm_head 非 Parameter'
        return 'self.lm_head.t(), None          # Parameter(D,V) -> (V,D) 视图，梯度回传统一 Parameter'
    if 'self.head' in expr:
        seg = src.split('self.head =')[-1][:80] if 'self.head =' in src else ''
        return 'self.head.weight, None' if 'bias=False' in seg else 'self.head.weight, getattr(self.head, "bias", None)'
    raise ValueError('unknown head: ' + expr)


def build(lines, fname):
    r = find_target_forward(lines)
    assert r, f'{fname}: 未找到含 head 投影的 forward'
    dstart, pname, body, proj, expr, ind = r
    cs, ce = class_bounds(lines, dstart)
    hi, hr = hidden_expr(expr), head_ret(expr, '\n'.join(lines))
    pad = ' ' * ind
    body_lines = []
    for i in body:
        if i >= proj:
            break
        l = lines[i]
        if l.strip() == '' or re.match(r'\s*(loss|targets)\s*=', l) or l.strip().startswith('if targets'):
            continue
        body_lines.append(l)
    code = [f'    def forward_hidden(self, {pname}):',
            '        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""']
    code += body_lines + [f'{pad}return {hi}', '',
                          '    def head_params(self):',
                          '        """返回 (weight(V,D), bias(V,) 或 None)，与 forward 中 head 投影严格一致。"""',
                          f'        return {hr}', '']
    anchor = None
    for i in range(dstart, ce):
        if re.match(r'    def forward_logits\(self', lines[i]) or re.match(r'    def np\(self', lines[i]):
            anchor = i; break
    assert anchor is not None, f'{fname}: 类内插入锚点未找到 (class L{cs+1}-L{ce})'
    assert anchor > dstart, f'{fname}: 锚点在目标 forward 之前（异常）'
    new = '\n'.join(lines[:anchor]) + '\n' + '\n'.join(code) + '\n'.join(lines[anchor:])
    return new, hi, hr, pname, cs + 1, anchor + 1


def special_patch(src):
    anchor = '    def forward_logits(self, x):\n        return self.forward(x, None)[0]\n'
    assert src.count(anchor) == 1
    return src.replace(anchor, anchor + """
    def forward_hidden(self, x):
        \"\"\"蒸馏用：返回 head 投影之前的 hidden (B,T,D)（分块 KL 入口，兼容 tie 分支）。\"\"\"
        B, T = x.size(); D = self.D
        h = self.e(x).unsqueeze(1)
        h = self.ln(h)
        for _ in range(self.n_layer):
            h = bdh_full_layer(self, h, self.encoder, self.decoder, self.encoder_v, self.attn)
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        return h.view(B, T, D)

    def head_params(self):
        \"\"\"tie 时 head 权重即 embed.weight(V,D)；否则 lm_head(V,D) 转置视图。\"\"\"
        if self.tie:
            return self.e.weight, None
        return self.lm_head.t(), None
""", 1), 'h.view(B,T,D)', 'tie? e.weight : lm_head.t()'


for fname in FILES + SPECIAL:
    p = ROOT / fname
    bak = p.with_suffix('.py.bak')
    src = bak.read_text() if bak.exists() else p.read_text()   # 优先从备份恢复
    assert 'def forward_hidden' not in src, f'{fname}: 备份不干净'
    try:
        if fname in SPECIAL:
            new, hi, hr = special_patch(src); pn = 'x'; cs, an = 0, 0
        else:
            new, hi, hr, pn, cs, an = build(src.split('\n'), fname)
    except Exception as e:
        print(f'[FAIL] {fname}: {e}')
        continue
    print(f'[ok]   {fname:32s} class@L{cs:<4} anchor@L{an:<4} hidden={hi:26s} head={hr.split("#")[0].strip()}')
    if APPLY:
        if not bak.exists():
            bak.write_text(src)
        p.write_text(new)
print('\nAPPLY' if APPLY else '\nDRY-RUN')
