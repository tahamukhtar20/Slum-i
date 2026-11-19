"""Counts quoted in the Results section that the other scripts do not print: settings where Ours or
the baseline beats the supervised model trained on all labels in both metrics, and every setting where
Ours lowers the mean slum IoU or mIoU. Reads results/all.json. Usage: python analysis/results_counts.py"""
import json
from statistics import mean

D = json.load(open("results/all.json"))
SEEDS, SATS, CITIES, BUDGETS = [0, 42, 123, 999, 1337], ["esri", "s2"], ["lahore", "karachi", "mumbai"], ["10", "20", "30"]
SUP = {"fixmatch": "sup_r101", "unimatch": "sup_r101", "unimatchv2": "sup_dino"}


def m(sat, c, meth, b, i):
    return mean(D[f"{sat}|{c}|{meth}|{b}|{s}"][i] for s in SEEDS)


beats = {"ours": 0, "baseline": 0}
drops = {0: [], 1: []}
for sat in SATS:
    for c in CITIES:
        for base, sup in SUP.items():
            full = [m(sat, c, sup, "100", i) for i in (0, 1)]
            for b in BUDGETS:
                for key, meth in [("ours", base + "_freematchc"), ("baseline", base)]:
                    if all(m(sat, c, meth, b, i) > full[i] for i in (0, 1)):
                        beats[key] += 1
                for i in (0, 1):
                    d = m(sat, c, base + "_freematchc", b, i) - m(sat, c, base, b, i)
                    if d < 0:
                        drops[i].append((round(d, 2), sat, c, base, b))
print("settings beating the all-label supervised model in both metrics:", beats)
print(f"slum IoU drops ({len(drops[1])}):", sorted(drops[1]))
print(f"mIoU drops ({len(drops[0])}):", sorted(drops[0]))
