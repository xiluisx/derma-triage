import timm
import torch.nn as nn


def create_model(name: str, pretrained: bool = True, num_classes: int = 3) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
