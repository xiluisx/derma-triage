// Orden = contrato con ml/constants.py y con la salida del ONNX.
export const CLASSES = ["benign", "malignant", "non-neoplastic"] as const;
export type ClassName = (typeof CLASSES)[number];

export const CLASS_LABELS_ES: Record<ClassName, string> = {
  benign: "Benigno",
  malignant: "Maligno",
  "non-neoplastic": "No neoplásico",
};

export const MALIGNANT_IDX = 1;
