"""Filtra las predicciones de una corrida CNN a la submuestra que vio Claude.

Uso: python -m ml.vlm.subset_cnn --cnn-run runs/<config>_f0 --subset runs/vlm_claude/test_index.csv
Salida: runs/cnn_vlm_subset/ (después: python -m ml.evaluate --run runs/cnn_vlm_subset)
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run", required=True)
    ap.add_argument("--subset", default="runs/vlm_claude/test_index.csv")
    ap.add_argument("--out", default="runs/cnn_vlm_subset")
    args = ap.parse_args()
    src, out = Path(args.cnn_run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep = set(pd.read_csv(args.subset).md5hash)
    test = pd.read_csv(src / "preds_test.csv")
    test[test.md5hash.isin(keep)].to_csv(out / "preds_test.csv", index=False)
    shutil.copy(src / "preds_val.csv", out / "preds_val.csv")  # la temperatura se ajusta igual, en val completo
    info = json.loads((src / "run.json").read_text())
    info.update({"run_id": "cnn_vlm_subset", "config": "cnn_vlm_subset", "subset_of": info["run_id"]})
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(f"{len(keep)} imágenes -> {out}")


if __name__ == "__main__":
    main()
