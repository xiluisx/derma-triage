"""Redimensiona a lado corto 256, guarda JPEG q95 y calcula phash.

Uso: python -m ml.data.prepare --raw data/raw --out data/processed
Salida: data/processed/images/<md5>.jpg, data/processed/meta.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.constants import CLASS_TO_IDX, PREPARED_SHORT_SIDE
from ml.preprocess import load_rgb


def process(args: tuple[str, str]) -> tuple[str, str | None]:
    src, dst = args
    try:
        img = load_rgb(src)
        w, h = img.size
        s = PREPARED_SHORT_SIDE / min(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
        img.save(dst, quality=95)
        return Path(dst).stem, str(imagehash.phash(img))
    except Exception:
        return Path(dst).stem, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(raw / "download_status.csv")
    status = status[status.downloaded]

    jobs = [(str(raw / "images" / f"{m}.jpg"), str(out / "images" / f"{m}.jpg")) for m in status.md5hash]
    with ProcessPoolExecutor() as ex:
        results = dict(tqdm(ex.map(process, jobs, chunksize=32), total=len(jobs), desc="prepare"))

    status["phash"] = status.md5hash.map(results)
    status = status[status.phash.notna()]
    meta = pd.DataFrame({
        "md5hash": status.md5hash,
        "label_idx": status.three_partition_label.map(CLASS_TO_IDX).astype(int),
        "tone": status.fitzpatrick_scale.astype(int),
        "phash": status.phash,
    })
    meta.to_csv(out / "meta.csv", index=False)
    print(f"Preparadas: {len(meta)}")


if __name__ == "__main__":
    main()
