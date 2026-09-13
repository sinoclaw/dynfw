"""生成 DynFW 全局状态图（2x2）—— 数据全部来自 docs/ARCHITECTURE-MAP.md 实测台账。
明确区分【实测】与【预估】，预估项用虚线/浅色标注。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130
plt.rcParams['savefig.dpi'] = 130

fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
fig.suptitle('DynFW 全局状态（截至 2026-09-13，T=8192 / batch=1）', fontsize=17, fontweight='bold')

REAL = '#1f6feb'      # 实测
EST  = '#b8c4d9'      # 预估
TF_C = '#d1495b'
V67_C = '#1f6feb'
V66_C = '#8e7cc3'
V6_C  = '#e0a458'

# ---------- ① 显存构成瀑布 ----------
ax = axes[0, 0]
labels = ['base\n(实测)', 'grad_ckpt\n(-30.4% 实测)', 'rope_fast\n(预估)', 'share_qk\n(预估)', 'bf16\n(预估)', 'TF\n(对照)']
vals   = [4.746, 3.302, 3.05, 2.92, 1.50, 0.634]
colors = [REAL, REAL, EST, EST, EST, TF_C]
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.8)
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.08, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
# 标注省下的部分
for i in range(1, len(vals)-1):
    ax.annotate('', xy=(i, vals[i]), xytext=(i-1, vals[i-1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.4, ls='--'))
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('训练峰值显存 (GiB)'); ax.set_title('① 显存：谁省的、省了多少', fontsize=13, fontweight='bold')
ax.axhline(0.634, color=TF_C, ls=':', lw=1.2)
ax.text(0.99, 0.70, 'TF 基线 0.634 GiB', color=TF_C, fontsize=9.5, ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='round', fc='white', ec=TF_C, alpha=0.9))
ax.grid(axis='y', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=REAL, label='已实测'), Patch(facecolor=EST, label='预估(待GPU验证)'),
                   Patch(facecolor=TF_C, label='TF 对照')], fontsize=9, loc='upper right')

# ---------- ② 三方能力对比 ----------
ax = axes[0, 1]
names = ['v6.7\n(FLA)', 'v6.6\n(手写)', 'v6+opt5', 'TF\n(对照)']
loss  = [5738.0, 7142.6, 8260.9, 19734.9]
stds  = [21.3, 80.4, 92.1, 43.6]
cols  = [V67_C, V66_C, V6_C, TF_C]
b = ax.bar(names, loss, color=cols, edgecolor='black', linewidth=0.8, yerr=stds, capsize=5)
for bi, v in zip(b, loss):
    ax.text(bi.get_x()+bi.get_width()/2, v+350, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('final_loss（越低越好）\nWelch: v6.7 vs TF t=-406.6 p<1e-6')
ax.set_title('② 能力：v6.7 赢 TF 70.9%（+3 seed 误差棒）', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# ---------- ③ 扫 T 速度比（交叉点） ----------
ax = axes[1, 0]
T   = [8192, 16384, 32768]
v67 = [51.3, 101.1, 200.5]     # v6.7 + compile
tf  = [45.7, 165.6, 629.5]     # TF
ratio = [a/b_ for a, b_ in zip(v67, tf)]
x = np.arange(len(T)); w = 0.36
ax.bar(x - w/2, v67, w, label='v6.7 + compile', color=V67_C, edgecolor='black', linewidth=0.7)
ax.bar(x + w/2, tf,  w, label='TF (flash attn)', color=TF_C, edgecolor='black', linewidth=0.7)
for i, r in enumerate(ratio):
    ax.text(i, max(v67[i], tf[i]) + 22, f'{r:.2f}×', ha='center', fontsize=11, fontweight='bold',
            color='green' if r < 1 else 'red')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'T={t//1024}K' for t in T], fontsize=11)
ax.set_ylabel('fwd+bwd 耗时 (ms)')
ax.set_title('③ 速度：交叉点在 T=8K~16K 之间，长 T 反超', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.text(0.98, 0.72, '比值 <1 = v6.7 更快\n比值 >1 = TF 更快', transform=ax.transAxes, ha='right',
        fontsize=9.5, style='italic', bbox=dict(boxstyle='round', fc='#f0f4ff', ec='#b8c4d9'))

# ---------- ④ 深度消融设计矩阵 ----------
ax = axes[1, 1]
ax.axis('off')
ax.set_title('④ 下一步：深度消融设计（待跑，回答"靠步数还是靠层数"）', fontsize=13, fontweight='bold')
tbl = [
    ['维度', '参数成本', '当前', '性质'],
    ['n_layer 层数', '+3,146,241/层', '2', '真分层（付费）'],
    ['steps 循环', '+0（复用参数）', '1 ← 未启用', '白送深度（免费）'],
    ['TF 每层对照', '+196,608/层', '2', '我们每层贵 16×'],
]
t = ax.table(cellText=tbl, loc='upper center', cellLoc='left', colWidths=[0.26, 0.27, 0.2, 0.27])
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 2.05)
for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1f6feb'); cell.set_text_props(color='white', fontweight='bold')
    elif r == 2 and c == 2:
        cell.set_facecolor('#ffe066')
ax.text(0.02, 0.30, '参数账（已核实，与实测 45,188,098 精确吻合）：', transform=ax.transAxes,
        fontsize=11, fontweight='bold')
ax.text(0.02, 0.21, '  embedding + head = 38,895,616（86%）  ← Qwen 词表 151936 带来的固定开销',
        transform=ax.transAxes, fontsize=10.5)
ax.text(0.02, 0.13, '  2 层计算 = 6,292,482（14%）        ← 真正的算力密度只有 14%',
        transform=ax.transAxes, fontsize=10.5, color='#d1495b')
ax.text(0.02, 0.03, '注：全库无 steps/n_layer 消融数据 —— 该矩阵未跑，不做结论',
        transform=ax.transAxes, fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.965])
out = '/data/dynfw/docs/dynfw-status-20260913.png'
plt.savefig(out, bbox_inches='tight', facecolor='white')
print('saved:', out)
import os
print('size:', os.path.getsize(out) // 1024, 'KB')
