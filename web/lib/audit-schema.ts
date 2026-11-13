import { z } from "zod";

// Réplica de ml/schema.py (fuente de verdad). .strict() = extra="forbid".
const MeanStd = z.object({ mean: z.number(), std: z.number() }).strict();

const ExperimentSummary = z
  .object({
    config: z.string(),
    display_name: z.string(),
    source: z.enum(["cnn", "vlm"]),
    n_runs: z.number().int(),
    macro_f1: MeanStd,
    auroc: MeanStd.nullable(),
    malignant_recall: MeanStd,
    ece: MeanStd.nullable(),
    malignant_recall_by_group: z.record(z.string(), MeanStd),
    macro_f1_by_group: z.record(z.string(), MeanStd),
    malignant_recall_gap: MeanStd,
  })
  .strict();

const QuantGate = z
  .object({
    variant: z.enum(["int8", "fp16"]),
    passed: z.boolean(),
    size_mb: z.number(),
    macro_f1_fp32: z.number(),
    macro_f1_quant: z.number(),
    macro_f1_drop: z.number(),
    malignant_recall_drop_by_group: z.record(z.string(), z.number()),
  })
  .strict();

const Comparison = z
  .object({
    config_a: z.string(),
    config_b: z.string(),
    metric: z.string(),
    diff: z.number(),
    ci_low: z.number(),
    ci_high: z.number(),
  })
  .strict();

export const AuditSummary = z
  .object({
    schema_version: z.literal(1),
    generated_at: z.string(),
    groups: z.array(z.string()),
    experiments: z.array(ExperimentSummary),
    quantization: z.array(QuantGate),
    comparisons: z.array(Comparison).default([]),
  })
  .strict();

export type AuditSummary = z.infer<typeof AuditSummary>;
export type ExperimentSummary = z.infer<typeof ExperimentSummary>;
export type QuantGate = z.infer<typeof QuantGate>;
export type Comparison = z.infer<typeof Comparison>;
