"""Diferencia pareada b - a sobre las mismas imágenes, con IC bootstrap 95%.

Uso: python -m ml.compare --a runs/convnext_base_cv --b runs/<mejor>_cv --out runs/comparisons.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.constants import GROUP_NAMES, MALIGNANT_IDX, tone_group

LOGITS = ["logit_0", "logit_1", "logit_2"]
N_BOOT = 2000


def recall_by_group(y, pred, groups) -> dict[str, float]:
    out = {}
    for g in GROUP_NAMES:
        m = (groups == g) & (y == MALIGNANT_IDX)
        out[g] = (pred[m] == MALIGNANT_IDX).mean() if m.any() else np.nan
    return out


def metrics(y, pred, groups) -> dict[str, float]:
    r = recall_by_group(y, pred, groups)
    vals = [v for v in r.values() if not np.isnan(v)]
    return {"macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            **{f"malignant_recall_{g}": v for g, v in r.items()}, "malignant_recall_gap": max(vals) - min(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="runs/comparisons.json")
    args = ap.parse_args()
    a = pd.read_csv(Path(args.a) / "preds_test.csv").set_index("md5hash")
    b = pd.read_csv(Path(args.b) / "preds_test.csv").set_index("md5hash").loc[a.index]  # mismas imágenes, mismo orden

    y = a.label_idx.to_numpy()
    groups = a.tone.map(tone_group).to_numpy()
    pa, pb = a[LOGITS].to_numpy().argmax(1), b[LOGITS].to_numpy().argmax(1)
    point = {k: metrics(y, pb, groups)[k] - v for k, v in metrics(y, pa, groups).items()}

    rng = np.random.default_rng(0)
    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        ma, mb = metrics(y[idx], pa[idx], groups[idx]), metrics(y[idx], pb[idx], groups[idx])
        for k in point:
            boots[k].append(mb[k] - ma[k])

    name_a, name_b = Path(args.a).name, Path(args.b).name
    results = []
    for k, d in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        results.append({"config_a": name_a, "config_b": name_b, "metric": k,
                        "diff": float(d), "ci_low": float(lo), "ci_high": float(hi)})
        print(f"{k:28s} {d:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  {'✔ significativo' if lo > 0 or hi < 0 else '· no concluyente'}")
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
