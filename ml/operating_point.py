"""CNN vs VLM a igual tasa de falsas alarmas de "maligno".

Los VLM devuelven una etiqueta (un solo punto de operación); la CNN da probabilidades, así que su umbral de
"maligno" se puede mover. Para comparar sensibilidad de forma justa se fija la CNN a la MISMA tasa de falsas
alarmas (FPR: no-malignos predichos como malignos) que cada VLM y se compara el recall maligno.

Regla de decisión de la CNN con umbral t: "maligno" si P(maligno) >= t; si no, argmax entre benigno/no neoplásico.
El umbral se elige en VALIDACIÓN (fold 0) para igualar la FPR del VLM y luego se aplica al test: así el test no se
usa para ajustar nada. Como control también se reporta el umbral "oráculo" ajustado sobre el propio test.

Uso: python -m ml.operating_point --cnn-run runs/convnext_base_f0 --subset runs/cnn_vlm_subset \
         --vlms runs/vlm_claude runs/vlm_medgemma --out runs/operating_point.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.constants import GROUP_NAMES, MALIGNANT_IDX, tone_group
from ml.metrics import softmax

LOGITS = ["logit_0", "logit_1", "logit_2"]
N_BOOT = 2000


def cnn_probs(df: pd.DataFrame, t: float) -> np.ndarray:
    return softmax(df[LOGITS].to_numpy() / t)


def predict_at(probs: np.ndarray, thr: float) -> np.ndarray:
    others = probs.copy()
    others[:, MALIGNANT_IDX] = -1
    return np.where(probs[:, MALIGNANT_IDX] >= thr, MALIGNANT_IDX, others.argmax(1))


def fpr(y: np.ndarray, pred: np.ndarray) -> float:
    neg = y != MALIGNANT_IDX
    return float((pred[neg] == MALIGNANT_IDX).mean())


def recall(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == MALIGNANT_IDX
    return float((pred[pos] == MALIGNANT_IDX).mean()) if pos.any() else float("nan")


def threshold_for_fpr(y: np.ndarray, p_mal: np.ndarray, target: float) -> float:
    """Umbral más bajo cuya FPR no excede target (cuantil de P(maligno) entre los no malignos)."""
    neg = np.sort(p_mal[y != MALIGNANT_IDX])[::-1]
    k = int(np.floor(target * len(neg)))  # nº de falsas alarmas permitidas
    return float(neg[k]) + 1e-12 if k < len(neg) else 0.0


def by_group(y, pred, groups) -> dict:
    out = {}
    for g in GROUP_NAMES:
        m = groups == g
        out[g] = {"recall": recall(y[m], pred[m]), "fpr": fpr(y[m], pred[m]),
                  "n_malignant": int((y[m] == MALIGNANT_IDX).sum()), "n_negative": int((y[m] != MALIGNANT_IDX).sum())}
    return out


def paired_recall_diff(y, pa, pb, groups) -> dict:
    """IC 95% bootstrap pareado de recall(b) - recall(a), global y por grupo."""
    rng = np.random.default_rng(0)
    keys = ["overall", *GROUP_NAMES]
    boots = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        yy, a, b, gg = y[i], pa[i], pb[i], groups[i]
        boots["overall"].append(recall(yy, b) - recall(yy, a))
        for g in GROUP_NAMES:
            m = gg == g
            boots[g].append(recall(yy[m], b[m]) - recall(yy[m], a[m]))
    point = {"overall": recall(y, pb) - recall(y, pa)}
    for g in GROUP_NAMES:
        m = groups == g
        point[g] = recall(y[m], pb[m]) - recall(y[m], pa[m])
    res = {}
    for k in keys:
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        res[k] = {"diff": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run", default="runs/convnext_base_f0")
    ap.add_argument("--subset", default="runs/cnn_vlm_subset")
    ap.add_argument("--vlms", nargs="+", default=["runs/vlm_claude", "runs/vlm_medgemma"])
    ap.add_argument("--out", default="runs/operating_point.json")
    args = ap.parse_args()

    run = Path(args.cnn_run)
    t = json.loads((run / "temperature.json").read_text())["temperature"]
    val = pd.read_csv(run / "preds_val.csv")
    p_val = cnn_probs(val, t)
    y_val = val.label_idx.to_numpy()

    sub = pd.read_csv(Path(args.subset) / "preds_test.csv").set_index("md5hash")
    p_sub = softmax(sub[LOGITS].to_numpy()) if json.loads((Path(args.subset) / "run.json").read_text()).get(
        "pre_calibrated") else cnn_probs(sub.reset_index(), t)
    y = sub.label_idx.to_numpy()
    groups = sub.tone.map(tone_group).to_numpy()
    pred_default = p_sub.argmax(1)

    results = {"cnn_default": {"fpr": fpr(y, pred_default), "recall": recall(y, pred_default),
                               "by_group": by_group(y, pred_default, groups)}, "matched": []}
    print(f"CNN (argmax)            FPR={fpr(y, pred_default):.3f}  recall={recall(y, pred_default):.3f}")

    for vdir in map(Path, args.vlms):
        v = pd.read_csv(vdir / "preds_test.csv").set_index("md5hash").loc[sub.index]
        pv = v.pred_idx.to_numpy()
        target = fpr(y, pv)
        thr_val = threshold_for_fpr(y_val, p_val[:, MALIGNANT_IDX], target)   # honesto: ajustado en validación
        thr_orc = threshold_for_fpr(y, p_sub[:, MALIGNANT_IDX], target)       # oráculo: ajustado en el test
        pc = predict_at(p_sub, thr_val)
        po = predict_at(p_sub, thr_orc)
        diff = paired_recall_diff(y, pv, pc, groups)  # CNN(umbral) - VLM
        row = {
            "vlm": vdir.name, "vlm_fpr": target, "vlm_recall": recall(y, pv), "vlm_by_group": by_group(y, pv, groups),
            "cnn_threshold_val": thr_val, "cnn_fpr": fpr(y, pc), "cnn_recall": recall(y, pc),
            "cnn_by_group": by_group(y, pc, groups),
            "cnn_threshold_oracle": thr_orc, "cnn_oracle_fpr": fpr(y, po), "cnn_oracle_recall": recall(y, po),
            "recall_diff_cnn_minus_vlm": diff,
        }
        results["matched"].append(row)
        print(f"\n== vs {vdir.name}: FPR objetivo {target:.3f}")
        print(f"  {vdir.name:14s} FPR={target:.3f} recall={row['vlm_recall']:.3f}  "
              + " ".join(f"{g}:{row['vlm_by_group'][g]['recall']:.2f}" for g in GROUP_NAMES))
        print(f"  CNN umbral={thr_val:.3f} FPR={row['cnn_fpr']:.3f} recall={row['cnn_recall']:.3f}  "
              + " ".join(f"{g}:{row['cnn_by_group'][g]['recall']:.2f}" for g in GROUP_NAMES))
        print(f"  (oráculo) umbral={thr_orc:.3f} FPR={row['cnn_oracle_fpr']:.3f} recall={row['cnn_oracle_recall']:.3f}")
        for k, d in diff.items():
            sig = "✔" if d["ci_low"] > 0 or d["ci_high"] < 0 else "·"
            print(f"  Δrecall CNN−VLM {k:7s} {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {sig}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
