"""Constantes compartidas. El orden de CLASSES es un contrato con la app web."""

CLASSES = ["benign", "malignant", "non-neoplastic"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MALIGNANT_IDX = 1

# Grupos de tono de piel (escala Fitzpatrick) para la auditoría.
TONE_GROUPS = {"I-II": (1, 2), "III-IV": (3, 4), "V-VI": (5, 6)}
GROUP_NAMES = list(TONE_GROUPS.keys())
UNKNOWN_GROUP = "unknown"

IMAGE_SIZE = 224
PREPARED_SHORT_SIDE = 256  # las imágenes del dataset se guardan con este lado corto

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

UNCERTAINTY_THRESHOLD = 0.6  # prob. calibrada máxima por debajo de esto => "incierto"

DISPLAY_NAMES = {
    "effnet_base": "EfficientNet-B0 (baseline)",
    "convnext_base": "ConvNeXt-T",
    "convnext_reweight": "ConvNeXt-T + reweighting",
    "convnext_coloraug": "ConvNeXt-T + color aug",
    "convnext_both": "ConvNeXt-T + ambas",
    "cnn_vlm_subset": "CNN elegida (submuestra de Claude)",
    "vlm_claude": "Claude (zero-shot)",
    "vlm_medgemma": "MedGemma 4B (zero-shot)",
}


def tone_group(tone: int) -> str:
    for name, tones in TONE_GROUPS.items():
        if tone in tones:
            return name
    return UNKNOWN_GROUP
