"""Collect the test scores of every run in scripts/runs/*.txt into results/runs.csv and results/all.json.

Scores are those of the model after the last epoch ("latest", not the EMA model and not the best
validation checkpoint), as in the paper. Usage, from the repo root after training:
    python analysis/collect_results.py [--root .]
"""
import argparse, csv, json
from pathlib import Path

LABELS = {  # run folder -> name in the paper
    "sup_r101": "Supervised (R101)", "fixmatch": "FixMatch", "fixmatch_freematchc": "Ours (FixMatch)",
    "unimatch": "UniMatch", "unimatch_freematchc": "Ours (UniMatch)", "sup_dino": "Supervised (DINOv2)",
    "unimatchv2": "UniMatch V2", "unimatchv2_freematchc": "Ours (UniMatch V2)",
    "unimatchv2_rank": "UniMatch V2 + DWL ranking", "freematch": "UniMatch V2 + FreeMatch",
    "softmatch": "UniMatch V2 + SoftMatch", "cdmad": "UniMatch V2 + CDMAD",
    "satglobal": "Ours, global term only", "fixedcls": "Ours, fixed class thresholds",
}
FIELDS = ["table", "sensor", "city", "method", "paper_name", "budget", "seed", "miou", "slum_iou",
          "background_iou", "accuracy", "config", "run_dir"]


def parse(save):
    p = Path(save).parts  # runs/exp_final/<sensor>/<city>/<method>/<budget>/seed<s>
    if p[1] == "exp_final":
        return p[2], p[3], p[4], p[5], int(p[6][4:])
    return "esri", p[3], p[4], p[5], int(p[6][4:])  # runs/exp_v2/{plug,abl}/<city>/<method>/10/seed<s>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder that holds runs/")
    a = ap.parse_args()
    rows, missing = [], 0
    for table in ["table2", "table3"]:
        for line in Path(f"scripts/runs/{table}.txt").read_text().splitlines():
            meth, cfg, _, save, *_ = line.split()
            sensor, city, method, budget, seed = parse(save)
            f = Path(a.root) / save / "results.json"
            if not f.exists():
                missing += 1
                continue
            r = json.loads(f.read_text())["latest"]["model"]
            rows.append(dict(table=table, sensor=sensor, city=city, method=method, paper_name=LABELS[method],
                             budget=budget, seed=seed, miou=r["macro_iou"], slum_iou=r["fg_iou"],
                             background_iou=r["bg_iou"], accuracy=r["accuracy"], config=cfg, run_dir=save))
    Path("results").mkdir(exist_ok=True)
    with open("results/runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    # all.json: the Table II runs keyed "<sensor>|<city>|<method>|<budget>|<seed>" -> [mIoU, slum IoU].
    all_ = {f"{r['sensor']}|{r['city']}|{r['method']}|{r['budget']}|{r['seed']}": [r["miou"], r["slum_iou"]]
            for r in rows if r["table"] == "table2"}
    Path("results/all.json").write_text(json.dumps(all_, indent=0, sort_keys=True))
    print(f"{len(rows)} runs collected, {missing} without results.json")


if __name__ == "__main__":
    main()
