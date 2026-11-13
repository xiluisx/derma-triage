"""Recall maligno por tono con IC 95% de las corridas agrupadas de CV → JSON estático para la vista Auditoría.

summary.json (ml.aggregate) solo trae media/std por grupo; los IC bootstrap viven en runs/<config>_cv/metrics_test.json.
Uso: python -m ml.export_cv_groups --runs runs --configs convnext_base convnext_coloraug \
         --out web/public/audit/cv_groups.json
Esquema (zod): web/lib/cv-groups-schema.ts
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--configs", nargs="+", default=["convnext_base", "convnext_coloraug"])
    ap.add_argument("--out", default="web/public/audit/cv_groups.json")
    args = ap.parse_args()

    runs, n = {}, None
    for c in args.configs:
        m = json.loads((Path(args.runs) / f"{c}_cv" / "metrics_test.json").read_text())
        n = m["overall"]["n"]
        gap = m["malignant_recall_gap"]
        runs[f"{c}_cv"] = {
            "by_group": {
                g: {"n": v["n"], "n_malignant": v["n_malignant"], "recall": v["malignant_recall"]["value"],
                    "ci_low": v["malignant_recall"]["ci_low"], "ci_high": v["malignant_recall"]["ci_high"]}
                for g, v in m["by_group"].items()
            },
            "gap": gap["value"], "gap_ci_low": gap["ci_low"], "gap_ci_high": gap["ci_high"],
        }
    Path(args.out).write_text(json.dumps({"schema_version": 1, "n": n, "runs": runs}, indent=2) + "\n")
    print(f"{len(runs)} corridas -> {args.out}")


if __name__ == "__main__":
    main()
