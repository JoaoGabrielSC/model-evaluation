from .embedding_strategy import EmbeddingStrategy
from .clip_strategy import ClipStrategy
from .huggingface_strategy import HuggingFaceStrategy
from .torchvision_strategy import TorchVisionStrategy

_STRATEGY_REGISTRY = {
    "clip": ClipStrategy,
    "vit": HuggingFaceStrategy,
    "deit": HuggingFaceStrategy,
    "dino": HuggingFaceStrategy,
    "resnet50": TorchVisionStrategy,
    "resnet18": TorchVisionStrategy,
    "effnet_b0": TorchVisionStrategy,
    "mobilenet": TorchVisionStrategy,
}


def get_strategy(model_name: str) -> EmbeddingStrategy:
    strategy_cls = _STRATEGY_REGISTRY.get(model_name.lower(), TorchVisionStrategy)
    return strategy_cls()
