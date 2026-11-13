"""Galería Grad-CAM offline: aciertos y errores en casos malignos por grupo de tono.

Usa el MISMO CAM que la app (ml.export.CamWrapper, con temperatura) y el mismo render que
web/lib/cam.ts (ReLU, normaliza por el máximo, bilineal a 224, turbo de 5 paradas, alfa = 200*v).
Las imágenes son de Fitzpatrick17k (CC BY-NC-SA): la salida NO va a git.

Uso: python -m ml.gradcam --run runs/convnext_base_f0 --data data/processed --out reports/figures/gradcam --per-cell 4
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.constants import CLASSES, GROUP_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, MALIGNANT_IDX, tone_group  # noqa: E402
from ml.export import build  # noqa: E402
from ml.preprocess import load_rgb, to_model_input  # noqa: E402

STOPS = [(0.0, (48, 18, 59)), (0.25, (70, 134, 251)), (0.5, (27, 229, 181)), (0.75, (250, 186, 57)), (1.0, (122, 4, 3))]
_T = np.array([s[0] for s in STOPS])
_C = np.array([s[1] for s in STOPS], dtype=np.float64)


def colormap(v: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(v, _T, _C[:, k]) for k in range(3)], -1).round()


def overlay(rgb: np.ndarray, cam_k: np.ndarray) -> np.ndarray:
    """rgb [224,224,3] float 0..1, cam_k [7,7] -> uint8 [224,224,3] (igual que cam.ts + alfa sobre la imagen)."""
    c = np.maximum(cam_k, 0)
    c = c / c.max() if c.max() > 0 else c
    # bilineal con align_corners=False y clamp de bordes, equivalente a cam.ts
    v = F.interpolate(torch.from_numpy(c)[None, None].float(), size=(IMAGE_SIZE, IMAGE_SIZE),
                      mode="bilinear", align_corners=False)[0, 0].numpy()
    a = (np.round(200 * v) / 255.0)[..., None]
    out = rgb * (1 - a) + colormap(v) / 255.0 * a
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="reports/figures/gradcam")
    ap.add_argument("--per-cell", type=int, default=4)
    args = ap.parse_args()
    run, data, out = Path(args.run), Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper, _ = build(run, dummy=False)
    wrapper = wrapper.to(dev).eval()

    preds = pd.read_csv(run / "preds_test.csv")
    preds["pred"] = preds[["logit_0", "logit_1", "logit_2"]].to_numpy().argmax(1)
    preds["group"] = preds.tone.map(tone_group)
    mal = preds[preds.label_idx == MALIGNANT_IDX]

    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    manifest = []
    for g in GROUP_NAMES:
        tiles = {True: [], False: []}
        for correct in (True, False):
            cell = mal[(mal.group == g) & ((mal.pred == MALIGNANT_IDX) == correct)]
            n_total = len(cell)
            for row in cell.head(args.per_cell).itertuples():
                x = to_model_input(load_rgb(data / "images" / f"{row.md5hash}.jpg"))
                with torch.no_grad():
                    probs, cam = wrapper(x[None].to(dev))
                probs, cam = probs[0].cpu().numpy(), cam[0].cpu().numpy()
                pred = int(probs.argmax())
                rgb = np.clip(x.permute(1, 2, 0).numpy() * std + mean, 0, 1)
                tag = "ok" if correct else "miss"
                f_pred = f"{g}_{tag}_{row.md5hash[:8]}.png"
                img_pred = overlay(rgb, cam[pred])
                Image.fromarray(img_pred).save(out / f_pred)
                entry = {"file": f_pred, "group": g, "correct": correct, "pred": CLASSES[pred],
                         "p_malignant": float(probs[MALIGNANT_IDX]), "md5hash": row.md5hash,
                         "n_in_cell": n_total}
                img_mal = img_pred
                if not correct:
                    f_mal = f"{g}_miss_{row.md5hash[:8]}_malcam.png"
                    img_mal = overlay(rgb, cam[MALIGNANT_IDX])
                    Image.fromarray(img_mal).save(out / f_mal)
                    entry["file_malignant_cam"] = f_mal
                manifest.append(entry)
                tiles[correct].append((img_pred, img_mal, entry))

        rows = [("Aciertos (CAM de la clase predicha)", [(t[0], t[2]) for t in tiles[True]]),
                ("Fallos (CAM de la clase predicha)", [(t[0], t[2]) for t in tiles[False]]),
                ("Fallos (CAM de maligno)", [(t[1], t[2]) for t in tiles[False]])]
        rows = [r for r in rows if r[1]]
        ncol = max(1, args.per_cell)
        fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.9 * len(rows)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for r, (title, items) in enumerate(rows):
            for c, (img, e) in enumerate(items):
                ax = axes[r][c]
                ax.imshow(img)
                ax.set_title(f"pred: {e['pred']}\nP(maligno)={e['p_malignant']:.2f}", fontsize=8)
            axes[r][0].text(-0.05, 0.5, title, transform=axes[r][0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=8)
        fig.suptitle(f"Fitzpatrick {g} - casos malignos de test", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"grid_{g}.png", dpi=150)
        plt.close(fig)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} imágenes -> {out}")


if __name__ == "__main__":
    main()
