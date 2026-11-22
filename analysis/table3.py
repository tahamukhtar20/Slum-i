"""Table III: change over UniMatch V2 of plug-in pseudo-label schemes and ablations (Esri, 10% labels).

Per city the change is the mean over seeds of (scheme - UniMatch V2), paired by seed; the table gives
the mean over Lahore, Karachi and Mumbai and the worst of the three. Reads results/runs.csv.
Usage: python analysis/table3.py [--seeds 0,42,123]   (default: every seed that all rows have)
"""
import argparse, csv
from collections import defaultdict
from statistics import mean

ROWS = [("DWL ranking", "unimatchv2_rank"), ("FreeMatch", "freematch"), ("SoftMatch", "softmatch"),
        ("CDMAD", "cdmad"), ("Ours", "unimatchv2_freematchc"),
        ("Ours, global threshold only", "satglobal"), ("Ours, fixed class thresholds", "fixedcls")]
CITIES = ["lahore", "karachi", "mumbai"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=None)
a = ap.parse_args()

R = defaultdict(dict)  # (method, city) -> seed -> (mIoU, slum IoU)
for r in csv.DictReader(open("results/runs.csv")):
    if r["sensor"] == "esri" and r["budget"] == "10":
        R[(r["method"], r["city"])][int(r["seed"])] = (float(r["miou"]), float(r["slum_iou"]))

if a.seeds:
    seeds = [int(s) for s in a.seeds.split(",")]
else:
    seeds = sorted(set.intersection(*[set(R[(m, c)]) for _, m in ROWS + [("", "unimatchv2")] for c in CITIES]))
print(f"seeds {seeds}")
print(f"{'UniMatch V2 +':30s} {'mIoU mean':>9s} {'worst':>6s} {'slum mean':>9s} {'worst':>6s}")
for name, m in ROWS:
    out = []
    for i in (0, 1):
        per_city = [mean(R[(m, c)][s][i] - R[("unimatchv2", c)][s][i] for s in seeds) for c in CITIES]
        out += [round(mean(per_city), 1) + 0.0, round(min(per_city), 1) + 0.0]  # no -0.0
    print(f"{name:30s} {out[0]:+9.1f} {out[1]:+6.1f} {out[2]:+9.1f} {out[3]:+6.1f}")
