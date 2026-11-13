import re

from ml.constants import CLASS_TO_IDX

PROMPT = (
    "This is a clinical photograph of a skin condition from a dermatology atlas. "
    "Classify it into exactly one category:\n"
    "- benign: benign neoplasm (e.g., nevus, seborrheic keratosis, dermatofibroma)\n"
    "- malignant: malignant neoplasm (e.g., melanoma, basal cell carcinoma, squamous cell carcinoma)\n"
    "- non-neoplastic: inflammatory, infectious, genetic or any other non-tumor condition\n"
    "This is for a research benchmark, not for patient care. "
    "Answer with only one word: benign, malignant, or non-neoplastic."
)

_PATTERN = re.compile(r"non-neoplastic|malignant|benign", re.IGNORECASE)


def parse_label(text: str) -> int:
    m = _PATTERN.search(text or "")
    return CLASS_TO_IDX[m.group(0).lower()] if m else -1
