"""Descarga Fitzpatrick17k y reporta disponibilidad por clase x tono.

Uso: python -m ml.data.download --out data/raw
Alternativa (espejo de Kaggle): python -m ml.data.download --from-zip "<ruta>.zip" --out data/raw
"""
import argparse
import hashlib
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research dataset download)"}


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            Image.open(io.BytesIO(r.content)).verify()  # descarta HTML/archivos corruptos
            dest.write_bytes(r.content)
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def ingest_zip(zip_path: Path, img_dir: Path) -> dict[str, bool]:
    """Copia <md5>.jpg del zip a img_dir. Válida = imagen legible Y md5 real del archivo == nombre."""
    ok: dict[str, bool] = {}
    mismatched = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        for name in tqdm(members, desc="zip"):
            md5 = Path(name).stem
            data = zf.read(name)
            if hashlib.md5(data).hexdigest() != md5:
                mismatched += 1
            try:
                Image.open(io.BytesIO(data)).verify()
            except Exception:
                ok[md5] = False
                continue
            (img_dir / f"{md5}.jpg").write_bytes(data)
            ok[md5] = True
    print(f"md5 distinto al nombre: {mismatched} de {len(members)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--from-zip", default=None, help="zip del espejo de Kaggle en vez de descargar de la web")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_URL)  # siempre el CSV oficial, no el del espejo
    df.to_csv(out / "fitzpatrick17k.csv", index=False)

    if args.from_zip:
        ok = ingest_zip(Path(args.from_zip), img_dir)
    else:
        df = df[df["url"].notna() & (df["url"] != "")]
        ok = {}
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch, row.url, img_dir / f"{row.md5hash}.jpg"): row.md5hash for row in df.itertuples()}
            for f in tqdm(as_completed(futs), total=len(futs), desc="download"):
                ok[futs[f]] = f.result()

    df["downloaded"] = df["md5hash"].map(ok).fillna(False).astype(bool)
    df.to_csv(out / "download_status.csv", index=False)

    table = pd.crosstab(df[df.downloaded]["three_partition_label"], df[df.downloaded]["fitzpatrick_scale"], margins=True)
    print(table)
    print(f"\nTotal disponibles: {int(df.downloaded.sum())} / {len(df)}")
    table.to_csv(out / "availability.csv")


if __name__ == "__main__":
    main()
