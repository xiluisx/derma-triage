"""Entrena 1 epoch sobre imágenes sintéticas en CPU. Evita perder horas de GPU por un typo."""
import numpy as np
import pandas as pd
from PIL import Image

from ml.train import run


def test_one_epoch_on_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(48):
        md5 = f"img{i:03d}"
        Image.fromarray(rng.integers(0, 255, (260, 300, 3), dtype=np.uint8)).save(img_dir / f"{md5}.jpg")
        tone = [1, 2, 3, 4, 5, 6, -1][i % 7]
        rows.append({"md5hash": md5, "label_idx": i % 3, "tone": tone, "fold": i % 5})
    pd.DataFrame(rows).to_csv(tmp_path / "splits.csv", index=False)

    cfg = {"name": "smoke", "model": "convnext_tiny", "pretrained": False, "epochs": 1, "batch_size": 8,
           "lr": 1e-4, "weight_decay": 0.05, "num_workers": 0, "amp": False, "device": "cpu",
           "tone_reweight": True, "color_aug": True}
    run_dir = run(cfg, fold=0, data_dir=tmp_path, out_root=tmp_path / "runs")

    for f in ["model.pt", "run.json", "history.json", "preds_val.csv", "preds_test.csv"]:
        assert (run_dir / f).exists(), f
    preds = pd.read_csv(run_dir / "preds_test.csv")
    assert len(preds) == 10 and {"logit_0", "logit_1", "logit_2"} <= set(preds.columns)  # fold 0 = 10 de 48
