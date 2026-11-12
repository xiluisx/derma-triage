import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.metrics import compute_run_metrics
from ml.schema import AuditSummary, RunMetrics


def test_example_contract_is_valid():
    AuditSummary.model_validate(json.loads(Path("contracts/audit-summary.example.json").read_text()))


def test_compute_run_metrics_produces_valid_schema():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"md5hash": [str(i) for i in range(n)], "label_idx": rng.integers(0, 3, n),
                       "tone": rng.choice([1, 2, 3, 4, 5, 6, -1], n)})
    probs = rng.dirichlet([1, 1, 1], n)
    m = compute_run_metrics(df, probs.argmax(1), probs, run_id="x_f0", config="x", seed=0, source="cnn", temperature=1.0)
    RunMetrics.model_validate_json(m.model_dump_json())
    assert set(m.by_group) == {"I-II", "III-IV", "V-VI"}
