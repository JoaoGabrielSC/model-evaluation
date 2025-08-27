from .embedding_strategy import EmbeddingStrategy
import torch
import numpy as np


class ClipStrategy(EmbeddingStrategy):
    def get_embedding(self, model, processor, image) -> list[float]:
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            return emb.flatten().tolist()
