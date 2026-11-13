import type { CropRect } from "../preprocess";

// "int8" solo existe en model-meta.json; no se sube a R2 (no pasó el quality gate) y no se usa.
export type Variant = "fp32" | "fp16" | "int8";

export interface ModelMeta {
  run_id: string;
  arch: string;
  classes: string[];
  input_size: number;
  temperature: number;
  uncertainty_threshold: number;
  cam_size: number;
  mean: number[];
  std: number[];
  variants: Partial<Record<Variant, string>>;
}

export interface Prediction {
  probs: number[];
  cam: Float32Array; // [3 * cam_size * cam_size]
  crop: CropRect;
  ms: number;
}

export type WorkerIn =
  | { type: "init"; url: string; variant: Variant; fallbackUrl?: string }
  | { type: "run"; id: number; bitmap: ImageBitmap };

export type WorkerOut =
  | { type: "progress"; loaded: number; total: number; cached?: boolean }
  | { type: "ready"; backend: string }
  | ({ type: "result"; id: number } & Prediction)
  | { type: "error"; id?: number; message: string };
