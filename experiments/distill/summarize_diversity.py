"""汇总「样本多样性隔离」实验：同 step(200)，唯一变量 = 输入多少段不同文本。

对照设计：
  A  5块×20ep  = 100 step（旧批，参考）
  B 40块×5ep   = 200 step，多样性 8×
  C  5块×40ep  = 200 step，多样性 1×
判读：v6 相对 TF 的劣势在 B 上显著小于 C ⇒ 落后主要来自"小样本拟合速度"；
      B 上仍显著落后 ⇒ 长上下文能力真短板。
"""
import glob
import json
import os
import statistics

R = '/data/dynfw/results'


def get(prefix):
    out = {}
    for d in sorted(glob.glob(os.path.join(R, prefix + '*'))):
        if not os.path.isdir(d) or 'smoke' in d:
            continue
        f = os.path.join(d, 'distill_result.json')
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        out[os.path.basename(d)] = j
    return out


B = get('diversity_')
print('=== 多样性实验（同 200 step：40块×5ep vs 5块×40ep）===')
rows = {}
for name, j in sorted(B.items()):
    arch = j['arch']
    tag = 'v6' if 'fw_cycle' in arch else 'tf'
    arm = 'B40' if 'B40' in name else 'C5'
    rows.setdefault(arm, {})[tag] = j
    print(f'  {name:24s} arch={arch:20s} loss={j["final_loss"]:12.3f} wall={j["wall_sec"]:7.1f}s '
          f'epochs={j["epochs"]} blocks={j["blocks"]}')

print()
for arm in ('B40', 'C5'):
    d = rows.get(arm, {})
    if 'v6' in d and 'tf' in d:
        v, t = d['v6']['final_loss'], d['tf']['final_loss']
        desc = '40块x5ep(多样性8x)' if arm == 'B40' else '5块x40ep(多样性1x)'
        print(f'  {arm} {desc:22s} v6={v:12.3f}  tf={t:12.3f}  v6/tf={v/t:.5f}  差={v-t:+10.3f}')
    else:
        print(f'  {arm} 数据不全（有 {list(d)}）')

print()
print('=== 参考：旧批（100 step, 5块x20ep）===')
for name in ('lt8192opt5_fw_s0', 'lt8192opt5_tf_s0'):
    f = os.path.join(R, name, 'distill_result.json')
    if os.path.exists(f):
        j = json.load(open(f))
        print(f'  {name:24s} loss={j["final_loss"]:12.3f} wall={j["wall_sec"]:7.1f}s '
              f'epochs={j["epochs"]} blocks={j["blocks"]}')
