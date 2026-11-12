"""Agrega runs/*/metrics_test.json -> summary.json (contrato AuditSummary).

Uso: python -m ml.aggregate --runs runs --out web/public/audit/summary.json
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.constants import DISPLAY_NAMES, GROUP_NAMES
from ml.schema import AuditSummary, Comparison, ExperimentSummary, MeanStd, QuantGate, RunMetrics


def display_name(config: str) -> str:
    if config.endswith("_cv"):
        return f"{DISPLAY_NAMES.get(config[:-3], config[:-3])} · CV 5 folds"
    return DISPLAY_NAMES.get(config, config)


def ms(values: list[float | None]) -> MeanStd | None:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return None
    return MeanStd(mean=float(np.mean(vals)), std=float(np.std(vals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="web/public/audit/summary.json")
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    by_config: dict[str, list[RunMetrics]] = defaultdict(list)
    for f in sorted(runs_dir.glob("*/metrics_test.json")):
        m = RunMetrics.model_validate_json(f.read_text())
        by_config[m.config].append(m)

    experiments = []
    for config, runs in by_config.items():
        experiments.append(ExperimentSummary(
            config=config,
            display_name=display_name(config),
            source=runs[0].source,
            n_runs=len(runs),
            macro_f1=ms([r.overall.macro_f1 for r in runs]),
            auroc=ms([r.overall.auroc for r in runs]),
            malignant_recall=ms([r.overall.malignant_recall for r in runs]),
            ece=ms([r.overall.ece for r in runs]),
            malignant_recall_by_group={g: ms([r.by_group[g].malignant_recall.value for r in runs]) for g in GROUP_NAMES},
            macro_f1_by_group={g: ms([r.by_group[g].macro_f1 for r in runs]) for g in GROUP_NAMES},
            malignant_recall_gap=ms([r.malignant_recall_gap.value for r in runs]),
        ))

    quant_path = runs_dir / "quant_gate.json"
    quant = [QuantGate.model_validate(q) for q in json.loads(quant_path.read_text())] if quant_path.exists() else []

    comp_path = runs_dir / "comparisons.json"
    comps = [Comparison.model_validate(c) for c in json.loads(comp_path.read_text())] if comp_path.exists() else []

    summary = AuditSummary(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                           groups=GROUP_NAMES, experiments=experiments, quantization=quant, comparisons=comps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary.model_dump_json(indent=2))
    for e in experiments:
        print(f"{e.display_name:32s} F1={e.macro_f1.mean:.3f}±{e.macro_f1.std:.3f}  "
              f"recall_mal={e.malignant_recall.mean:.3f}  gap={e.malignant_recall_gap.mean:.3f}")


if __name__ == "__main__":
    main()
