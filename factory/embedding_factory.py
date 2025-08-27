import torch
from torch import nn
from transformers import (
    ViTModel,
    ViTImageProcessor,
    CLIPModel,
    CLIPProcessor,
)
import torch
from torchvision import models, transforms
from config._types import HFModelConfig, TorchModelConfig, ModelConfig, Embedding


class EmbeddingFactory:
    MODELS: dict[str, ModelConfig] = {
        "vit": HFModelConfig(
            ViTModel, ViTImageProcessor, "google/vit-base-patch16-224-in21k", 768
        ),
        "clip": HFModelConfig(
            CLIPModel, CLIPProcessor, "openai/clip-vit-base-patch32", 512
        ),
        "dino": HFModelConfig(ViTModel, ViTImageProcessor, "facebook/dino-vits16", 384),
        "deit": HFModelConfig(
            ViTModel, ViTImageProcessor, "facebook/deit-base-patch16-224", 768
        ),
        "resnet50": TorchModelConfig(
            lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT), 2048
        ),
        "resnet18": TorchModelConfig(
            lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT), 512
        ),
        "effnet_b0": TorchModelConfig(
            lambda: models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            ),
            1280,
        ),
        "mobilenet": TorchModelConfig(
            lambda: models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            ),
            960,
        ),
    }

    @staticmethod
    def get(model_type: str) -> Embedding:
        model_type = model_type.lower()
        config = EmbeddingFactory.MODELS.get(model_type)

        if config is None:
            raise ValueError(f"Modelo '{model_type}' não suportado.")

        if isinstance(config, HFModelConfig):
            model = config.model_class.from_pretrained(
                config.model_name, use_safetensors=True, torch_dtype=torch.float32
            )

            processor = config.processor_class.from_pretrained(config.model_name)
            return Embedding(model, processor, config.dim, config.source)

        model = config.loader()
        model = nn.Sequential(*(list(model.children())[:-1]))
        processor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return Embedding(model, processor, config.dim, config.source)
