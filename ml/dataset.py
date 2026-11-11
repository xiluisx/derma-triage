import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ml.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TONE_GROUPS, tone_group
from ml.preprocess import load_rgb, to_model_input


def train_transform(color_aug: bool) -> v2.Compose:
    ops = [
        v2.PILToTensor(),
        v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(20),
    ]
    if color_aug:
        ops.append(v2.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.04))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir, train: bool, color_aug: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tf = train_transform(color_aug) if train else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = load_rgb(self.image_dir / f"{row.md5hash}.jpg")
        x = self.tf(img) if self.tf else to_model_input(img)
        return x, int(row.label_idx), i


def class_weights(labels: pd.Series, n_classes: int = 3) -> torch.Tensor:
    counts = labels.value_counts().reindex(range(n_classes), fill_value=1).to_numpy()
    w = 1.0 / counts**0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def tone_sample_weights(tones: pd.Series) -> torch.Tensor:
    groups = tones.map(tone_group)
    known = groups[groups.isin(TONE_GROUPS.keys())]
    counts = known.value_counts()
    n_known = len(known)
    w = groups.map(lambda g: n_known / (len(TONE_GROUPS) * counts[g]) if g in counts else 1.0)
    return torch.tensor(w.to_numpy(), dtype=torch.float32)
