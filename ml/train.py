"""Entrena un modelo y guarda predicciones de val/test.

Uso: python -m ml.train --config configs/convnext_base.yaml --fold 0 \
         --data-dir data/processed --out runs
Salida: runs/<name>_f<fold>/{model.pt, run.json, history.json, preds_val.csv, preds_test.csv}
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.data.split import load_split
from ml.dataset import SkinDataset, class_weights, tone_sample_weights
from ml.models import create_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def save_preds(df: pd.DataFrame, logits: np.ndarray, path: Path) -> None:
    out = df[["md5hash", "label_idx", "tone"]].reset_index(drop=True).copy()
    for k in range(logits.shape[1]):
        out[f"logit_{k}"] = logits[:, k]
    out.to_csv(path, index=False)


def run(cfg: dict, fold: int, data_dir: Path, out_root: Path, limit: int | None = None, seed: int = 0) -> Path:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu")
    amp = bool(cfg.get("amp", True))
    run_dir = out_root / f"{cfg['name']}_f{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split(data_dir, fold)
    if limit:
        splits = splits.groupby("split").head(limit)
    tr, va, te = (splits[splits.split == s] for s in ("train", "val", "test"))
    img_dir = data_dir / "images"

    nw = cfg.get("num_workers", 4)
    bs = cfg["batch_size"]
    dl_tr = DataLoader(SkinDataset(tr, img_dir, True, cfg.get("color_aug", False)), bs, shuffle=True,
                       num_workers=nw, pin_memory=True, drop_last=len(tr) > bs)
    dl_va = DataLoader(SkinDataset(va, img_dir, False), bs, num_workers=nw)
    dl_te = DataLoader(SkinDataset(te, img_dir, False), bs, num_workers=nw)

    model = create_model(cfg["model"], pretrained=cfg.get("pretrained", True)).to(device)
    cw = class_weights(tr.label_idx).to(device)
    sw = tone_sample_weights(tr.tone).to(device) if cfg.get("tone_reweight") else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(dl_tr)
    warm = cfg.get("warmup_epochs", 1) * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_f1, history = -1.0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for x, y, idx in tqdm(dl_tr, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, weight=cw, reduction="none")
                if sw is not None:
                    loss = loss * sw[idx.to(device)]
                loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val_logits = predict(model, dl_va, device, amp)
        f1 = f1_score(va.label_idx, val_logits.argmax(1), average="macro")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": float(f1), "secs": time.time() - t0})
        print(history[-1])
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), run_dir / "model.pt")

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    save_preds(va, predict(model, dl_va, device, amp), run_dir / "preds_val.csv")
    save_preds(te, predict(model, dl_te, device, amp), run_dir / "preds_test.csv")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "run.json").write_text(json.dumps({"run_id": run_dir.name, "config": cfg["name"], "seed": seed, "fold": fold,
                                                  "source": "cnn", "model": cfg["model"], "best_val_macro_f1": best_f1}, indent=2))
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--limit", type=int, default=None, help="filas por split (debug rápido)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(run(cfg, args.fold, Path(args.data_dir), Path(args.out), args.limit, args.seed))


if __name__ == "__main__":
    main()
