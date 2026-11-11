"""Deduplica por phash y asigna 5 folds estratificados por clase x tono.

Uso: python -m ml.data.split --data data/processed
Salida: data/processed/splits.csv  (md5hash,label_idx,tone,tone_group,phash,fold)

Para una corrida con fold k: test = fold k (20%); de lo demás se aparta val
(~15% del total, estratificado, semilla k); el resto es train (~65%).
El "split fijo" de exploración es simplemente fold 0. Usa siempre load_split().
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.constants import tone_group

DUP_MAX_HAMMING = 4
N_FOLDS = 5
VAL_FRAC_OF_REST = 0.15 / 0.80


def _strata(df: pd.DataFrame) -> pd.Series:
    return df.label_idx.astype(str) + "_" + df.tone.astype(str)


def load_split(data_dir: Path, fold: int = 0) -> pd.DataFrame:
    """splits.csv + columna `split` (train/val/test) para el fold indicado."""
    df = pd.read_csv(Path(data_dir) / "splits.csv")
    rest = df[df.fold != fold]
    try:
        _, val = train_test_split(rest, test_size=VAL_FRAC_OF_REST, stratify=_strata(rest), random_state=fold)
    except ValueError:  # estratos diminutos (solo pasa en datos sintéticos de prueba)
        _, val = train_test_split(rest, test_size=VAL_FRAC_OF_REST, random_state=fold)
    df["split"] = np.where(df.fold == fold, "test", np.where(df.md5hash.isin(val.md5hash), "val", "train"))
    return df


def phash_to_u64(h: str) -> np.uint64:
    return np.uint64(int(h, 16))


def near_duplicate_groups(hashes: np.ndarray, max_dist: int = DUP_MAX_HAMMING, chunk: int = 1024) -> np.ndarray:
    """Union-find sobre pares con distancia de Hamming <= max_dist. Devuelve id de grupo por fila."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        block = hashes[start : start + chunk]
        dist = np.bitwise_count(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(dist <= max_dist)
        for r, c in zip(rows + start, cols):
            if r < c:
                ra, rb = find(r), find(c)
                if ra != rb:
                    parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data)
    meta = pd.read_csv(data / "meta.csv").sort_values("md5hash").reset_index(drop=True)
    hashes = np.array([phash_to_u64(h) for h in meta.phash], dtype=np.uint64)
    meta["dup_group"] = near_duplicate_groups(hashes)

    # Grupos con etiquetas contradictorias se descartan completos (ambiguos).
    n_labels = meta.groupby("dup_group").label_idx.transform("nunique")
    conflicted = meta[n_labels > 1].dup_group.nunique()
    meta = meta[n_labels == 1]
    before = len(meta)
    meta = meta.drop_duplicates("dup_group", keep="first").drop(columns="dup_group")
    print(f"Duplicados eliminados: {before - len(meta)} · grupos en conflicto descartados: {conflicted}")

    meta["tone_group"] = meta.tone.map(tone_group)
    meta = meta.reset_index(drop=True)
    meta["fold"] = -1
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    for k, (_, test_idx) in enumerate(skf.split(meta, _strata(meta))):
        meta.loc[test_idx, "fold"] = k

    meta.to_csv(data / "splits.csv", index=False)
    print(pd.crosstab([meta.fold, meta.label_idx], meta.tone_group))


if __name__ == "__main__":
    main()
