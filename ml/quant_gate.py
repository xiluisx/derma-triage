"""Compara int8 y fp16 contra fp32 sobre el test completo, por grupo de tono.

Gate (spec §4): pérdida de macro-F1 <= 0.01 Y pérdida de recall de malignos <= 0.02 en cada grupo.
Uso: python -m ml.quant_gate --artifacts artifacts --data data/processed --out runs/quant_gate.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score
from tqdm import tqdm

from ml.constants import GROUP_NAMES, MALIGNANT_IDX, tone_group
from ml.data.split import load_split
from ml.preprocess import load_rgb, to_model_input
from ml.schema import QuantGate

MAX_F1_DROP = 0.01
MAX_RECALL_DROP = 0.02


def run_model(model: Path, xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Devuelve (probs [n,3], cam [n,3,7,7], ms por imagen)."""
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    probs, cams = [], []
    t0 = time.perf_counter()
    for x in tqdm(xs, desc=model.name):
        p, c = sess.run(["probs", "cam"], {"input": x})
        probs.append(p[0])
        cams.append(c[0])
    return np.array(probs), np.array(cams), (time.perf_counter() - t0) / len(xs) * 1000


def predict(model: Path, xs: list[np.ndarray]) -> np.ndarray:
    return run_model(model, xs)[0].argmax(1)


def recall_by_group(y, pred, groups) -> dict[str, float]:
    out = {}
    for g in GROUP_NAMES:
        m = (groups == g) & (y == MALIGNANT_IDX)
        out[g] = float((pred[m] == MALIGNANT_IDX).mean()) if m.any() else float("nan")
    return out


def gate(fp32_pred, q_pred, y, groups, variant: str, size_mb: float) -> QuantGate:
    f_ref = f1_score(y, fp32_pred, average="macro")
    f_q = f1_score(y, q_pred, average="macro")
    r_ref, r_q = recall_by_group(y, fp32_pred, groups), recall_by_group(y, q_pred, groups)
    drops = {g: r_ref[g] - r_q[g] for g in GROUP_NAMES}
    passed = (f_ref - f_q) <= MAX_F1_DROP and all(d <= MAX_RECALL_DROP for d in drops.values())
    return QuantGate(variant=variant, passed=passed, size_mb=size_mb, macro_f1_fp32=f_ref, macro_f1_quant=f_q,
                     macro_f1_drop=f_ref - f_q, malignant_recall_drop_by_group=drops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/quant_gate.json")
    ap.add_argument("--detail", default=None, help="JSON opcional con tabla detallada, paridad y latencia")
    args = ap.parse_args()
    art, data = Path(args.artifacts), Path(args.data)

    test = load_split(data).query("split == 'test'")
    xs = [to_model_input(load_rgb(data / "images" / f"{m}.jpg")).numpy()[None] for m in test.md5hash]
    y = test.label_idx.to_numpy()
    groups = test.tone.map(tone_group).to_numpy()

    p_ref, c_ref, ms_ref = run_model(art / "model-fp32.onnx", xs)
    ref = p_ref.argmax(1)
    results, detail = [], {}

    def row(pred):
        return {"macro_f1": float(f1_score(y, pred, average="macro")),
                "macro_f1_by_group": {g: float(f1_score(y[groups == g], pred[groups == g], average="macro"))
                                      for g in GROUP_NAMES},
                "malignant_recall_by_group": recall_by_group(y, pred, groups)}

    detail["fp32"] = {**row(ref), "ms_per_image": ms_ref}
    for v in ["int8", "fp16"]:
        path = art / f"model-{v}.onnx"
        p, c, ms = run_model(path, xs)
        results.append(gate(ref, p.argmax(1), y, groups, v, path.stat().st_size / 1e6))
        print(results[-1].model_dump_json(indent=2))
        detail[v] = {**row(p.argmax(1)), "ms_per_image": ms,
                     "max_prob_diff": float(np.abs(p - p_ref).max()),
                     "mean_prob_diff": float(np.abs(p - p_ref).mean()),
                     "argmax_disagreement": float((p.argmax(1) != ref).mean()),
                     "max_cam_diff": float(np.abs(c - c_ref).max()),
                     "mean_cam_diff": float(np.abs(c - c_ref).mean())}
    Path(args.out).write_text(json.dumps([r.model_dump() for r in results], indent=2))
    if args.detail:
        Path(args.detail).write_text(json.dumps(detail, indent=2))


if __name__ == "__main__":
    main()
