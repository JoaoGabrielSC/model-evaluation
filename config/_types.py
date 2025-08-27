from torch import nn
from transformers import (
    ViTModel,
    ViTImageProcessor,
    CLIPModel,
    CLIPProcessor,
)
import torch
from typing import Protocol
from typing import Protocol
from dataclasses import dataclass
from typing import Any, Union


class TorchModelLoader(Protocol):
    def __call__(self) -> nn.Module: ...


class PretrainedModel(Protocol):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs): ...


class PretrainedProcessor(Protocol):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs): ...


@dataclass
class HFModelConfig:
    model_class: PretrainedModel
    processor_class: PretrainedProcessor
    model_name: str
    dim: int
    source: str = "hf"


@dataclass
class TorchModelConfig:
    loader: TorchModelLoader
    dim: int
    source: str = "torchvision"


ModelConfig = Union[HFModelConfig, TorchModelConfig]


@dataclass
class Embedding:
    model: nn.Module
    processor: Any
    dim: int
    source: str
