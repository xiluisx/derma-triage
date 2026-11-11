"""Si hay fuga train->test, todas las métricas del proyecto son falsas."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.split import DUP_MAX_HAMMING, N_FOLDS, load_split, phash_to_u64

DATA = Path("data/processed")
SPLITS = DATA / "splits.csv"


@pytest.mark.skipif(not SPLITS.exists(), reason="corre python -m ml.data.split primero")
def test_no_near_duplicates_across_folds():
    df = pd.read_csv(SPLITS)
    assert df.md5hash.is_unique
    assert set(df.fold) == set(range(N_FOLDS))
    hashes = {k: np.array([phash_to_u64(h) for h in g.phash], dtype=np.uint64) for k, g in df.groupby("fold")}
    for k in range(N_FOLDS):
        others = np.concatenate([hashes[j] for j in range(N_FOLDS) if j != k])
        dist = np.bitwise_count(hashes[k][:, None] ^ others[None, :])
        assert dist.min() > DUP_MAX_HAMMING, f"casi-duplicados entre el fold {k} y el resto"


@pytest.mark.skipif(not SPLITS.exists(), reason="corre python -m ml.data.split primero")
def test_load_split_partitions_each_fold():
    for k in range(N_FOLDS):
        df = load_split(DATA, k)
        assert set(df[df.split == "test"].md5hash) == set(df[df.fold == k].md5hash)
        assert set(df.split) == {"train", "val", "test"}
        test = df[df.split == "test"]
        for g in ["I-II", "III-IV", "V-VI"]:
            assert ((test.tone_group == g) & (test.label_idx == 1)).sum() >= 10, (k, g)
