import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from ml.constants import GROUP_NAMES, MALIGNANT_IDX, tone_group
from ml.schema import CI, GroupMetrics, OverallMetrics, RunMetrics

N_BOOT = 1000


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: minimiza la NLL en validación optimizando log(T)."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def _safe_auroc(y: np.ndarray, probs: np.ndarray | None) -> float | None:
    if probs is None or len(np.unique(y)) < 3:
        return None
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def _malignant_recall(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == MALIGNANT_IDX
    return float((pred[pos] == MALIGNANT_IDX).mean()) if pos.any() else float("nan")


def _group_recalls(y, pred, groups) -> dict[str, float]:
    return {g: _malignant_recall(y[groups == g], pred[groups == g]) for g in GROUP_NAMES}


def compute_run_metrics(df: pd.DataFrame, pred: np.ndarray, probs: np.ndarray | None, *, run_id: str,
                        config: str, seed: int | None, source: str, temperature: float | None) -> RunMetrics:
    """df: md5hash,label_idx,tone. pred: índice predicho (-1 = abstención). probs: [n,3] o None."""
    y = df.label_idx.to_numpy()
    groups = df.tone.map(tone_group).to_numpy()
    labels = [0, 1, 2]

    overall = OverallMetrics(
        n=len(y),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        malignant_recall=_malignant_recall(y, pred),
        auroc=_safe_auroc(y, probs),
        ece=ece(probs, y) if probs is not None else None,
        abstentions=int((pred < 0).sum()),
    )

    rng = np.random.default_rng(0)
    boot_recalls = {g: [] for g in GROUP_NAMES}
    boot_gaps = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        r = _group_recalls(y[idx], pred[idx], groups[idx])
        for g in GROUP_NAMES:
            boot_recalls[g].append(r[g])
        vals = [v for v in r.values() if not np.isnan(v)]
        boot_gaps.append(max(vals) - min(vals) if len(vals) > 1 else np.nan)

    def ci(point: float, samples) -> CI:
        lo, hi = np.nanpercentile(samples, [2.5, 97.5])
        return CI(value=point, ci_low=float(lo), ci_high=float(hi))

    point = _group_recalls(y, pred, groups)
    by_group = {}
    for g in GROUP_NAMES:
        m = groups == g
        by_group[g] = GroupMetrics(
            n=int(m.sum()),
            n_malignant=int((y[m] == MALIGNANT_IDX).sum()),
            macro_f1=float(f1_score(y[m], pred[m], labels=labels, average="macro", zero_division=0)),
            malignant_recall=ci(point[g], boot_recalls[g]),
            auroc=_safe_auroc(y[m], probs[m] if probs is not None else None),
        )
    pv = [v for v in point.values() if not np.isnan(v)]
    gap = ci(max(pv) - min(pv), boot_gaps)

    return RunMetrics(run_id=run_id, config=config, seed=seed, source=source, temperature=temperature,
                      overall=overall, by_group=by_group, malignant_recall_gap=gap)
