"""preds -> temperature.json + metrics_test.json

CNN: python -m ml.evaluate --run runs/convnext_base_f0
VLM: python -m ml.evaluate --run runs/vlm_claude      (preds_test.csv con columna pred_idx)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ml.metrics import compute_run_metrics, fit_temperature, softmax


def evaluate_run(run_dir: Path) -> Path:
    info = json.loads((run_dir / "run.json").read_text())
    test = pd.read_csv(run_dir / "preds_test.csv")
    logit_cols = ["logit_0", "logit_1", "logit_2"]

    if info["source"] == "cnn" and info.get("pre_calibrated"):
        # Corrida agrupada de validación cruzada: logit_k ya es log(prob calibrada) de su fold.
        t, probs = None, softmax(test[logit_cols].to_numpy())
        pred = probs.argmax(1)
    elif info["source"] == "cnn":
        val = pd.read_csv(run_dir / "preds_val.csv")
        t = fit_temperature(val[logit_cols].to_numpy(), val.label_idx.to_numpy())
        (run_dir / "temperature.json").write_text(json.dumps({"temperature": t}))
        probs = softmax(test[logit_cols].to_numpy() / t)
        pred = probs.argmax(1)
    else:
        t, probs, pred = None, None, test.pred_idx.to_numpy()

    metrics = compute_run_metrics(test, pred, probs, run_id=info["run_id"], config=info["config"],
                                  seed=info.get("seed"), source=info["source"], temperature=t)
    out = run_dir / "metrics_test.json"
    out.write_text(metrics.model_dump_json(indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    out = evaluate_run(Path(args.run))
    m = json.loads(out.read_text())
    print(json.dumps({"overall": m["overall"], "gap": m["malignant_recall_gap"]}, indent=2))


if __name__ == "__main__":
    main()
