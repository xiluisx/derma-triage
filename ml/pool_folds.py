"""Junta las predicciones de test de los 5 folds en una corrida "<config>_cv".

Cada fold se calibra con SU temperatura y se guarda log(prob calibrada) como logit_k,
marcando la corrida como pre_calibrated (evaluate no vuelve a ajustar la temperatura).
Uso: python -m ml.pool_folds --config convnext_base --runs runs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.split import N_FOLDS
from ml.metrics import softmax

LOGITS = ["logit_0", "logit_1", "logit_2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    runs = Path(args.runs)

    parts = []
    for k in range(N_FOLDS):
        rd = runs / f"{args.config}_f{k}"
        t = json.loads((rd / "temperature.json").read_text())["temperature"]
        df = pd.read_csv(rd / "preds_test.csv")
        probs = softmax(df[LOGITS].to_numpy() / t)
        df[LOGITS] = np.log(np.clip(probs, 1e-12, 1.0))
        parts.append(df.assign(fold=k))
    pooled = pd.concat(parts, ignore_index=True)
    assert pooled.md5hash.is_unique, "una imagen aparece en más de un fold de test"

    out = runs / f"{args.config}_cv"
    out.mkdir(exist_ok=True)
    pooled.to_csv(out / "preds_test.csv", index=False)
    (out / "run.json").write_text(json.dumps({"run_id": out.name, "config": out.name, "seed": None, "source": "cnn",
                                              "pre_calibrated": True, "folds": N_FOLDS}, indent=2))
    print(f"{len(pooled)} predicciones -> {out}")


if __name__ == "__main__":
    main()
