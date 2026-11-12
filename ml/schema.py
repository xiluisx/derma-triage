from typing import Literal

from pydantic import BaseModel, ConfigDict


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CI(_M):
    value: float
    ci_low: float
    ci_high: float


class GroupMetrics(_M):
    n: int
    n_malignant: int
    macro_f1: float
    malignant_recall: CI
    auroc: float | None


class OverallMetrics(_M):
    n: int
    macro_f1: float
    malignant_recall: float
    auroc: float | None
    ece: float | None
    abstentions: int


class RunMetrics(_M):
    run_id: str
    config: str
    seed: int | None
    source: Literal["cnn", "vlm"]
    temperature: float | None
    overall: OverallMetrics
    by_group: dict[str, GroupMetrics]
    malignant_recall_gap: CI


class MeanStd(_M):
    mean: float
    std: float


class ExperimentSummary(_M):
    config: str
    display_name: str
    source: Literal["cnn", "vlm"]
    n_runs: int
    macro_f1: MeanStd
    auroc: MeanStd | None
    malignant_recall: MeanStd
    ece: MeanStd | None
    malignant_recall_by_group: dict[str, MeanStd]
    macro_f1_by_group: dict[str, MeanStd]
    malignant_recall_gap: MeanStd


class QuantGate(_M):
    variant: Literal["int8", "fp16"]
    passed: bool
    size_mb: float
    macro_f1_fp32: float
    macro_f1_quant: float
    macro_f1_drop: float
    malignant_recall_drop_by_group: dict[str, float]


class Comparison(_M):
    """Diferencia pareada (b - a) sobre las MISMAS imágenes, con IC bootstrap 95%."""
    config_a: str
    config_b: str
    metric: str  # p. ej. "malignant_recall_V-VI", "malignant_recall_gap"
    diff: float
    ci_low: float
    ci_high: float


class AuditSummary(_M):
    schema_version: Literal[1]
    generated_at: str
    groups: list[str]
    experiments: list[ExperimentSummary]
    quantization: list[QuantGate]
    comparisons: list[Comparison] = []
