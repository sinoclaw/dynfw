#!/usr/bin/env python3
"""汇总 corpus_en 正确口径批量对轰结果。"""
import glob, json, os, sys

base = "/data/dynfw/results"
rows = []
for jp in sorted(glob.glob(os.path.join(base, "*_corpusen_*", "distill_result.json"))):
    try:
        j = json.load(open(jp))
        rows.append({
            "dir": os.path.basename(os.path.dirname(jp)),
            "arch": j.get("arch","?"),
            "seed": j.get("seed","?"),
            "final": j.get("final_loss",float("nan")),
            "mean": j.get("mean_loss",float("nan")),
            "slots": j.get("slots","?"),
            "k": j.get("k","?"),
        })
    except Exception as e:
        print("ERR", jp, e)

# sort by (arch, seed)
print(f"{'arch':28} {'seed':>4} {'slots':>6} {'k':>4} {'final':>10} {'mean':>12}")
for r in sorted(rows, key=lambda x:(x["arch"], x["seed"])):
    print(f"{r['arch']:28} {str(r['seed']):>4} {str(r['slots']):>6} {str(r['k']):>4} {r['final']:>10.3f} {r['mean']:>12.3f}")

# Pivot final by arch -> list of seeds
arch_finals = {}
for r in rows:
    arch_finals.setdefault(r["arch"], []).append(r["final"])
print("\n=== per-arch final (final_loss) over seeds ===")
for a in sorted(arch_finals):
    fs = arch_finals[a]
    print(f"{a:28} n={len(fs)} finals={['%.3f'%f for f in fs]} min={min(fs):.3f} avg={sum(fs)/len(fs):.3f}")
