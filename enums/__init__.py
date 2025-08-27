from enum import StrEnum


class ModelType(StrEnum):
    VIT = "vit"
    CLIP = "clip"
    DINO = "dino"
    DEIT = "deit"
    RESNET50 = "resnet50"
    RESNET18 = "resnet18"
    EFFNET_B0 = "effnet_b0"
    MOBILENET = "mobilenet"
