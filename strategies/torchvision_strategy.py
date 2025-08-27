from .embedding_strategy import EmbeddingStrategy
import torch
import numpy as np
from typing import List


class TorchVisionStrategy(EmbeddingStrategy):
    def get_embedding(self, model, processor, image) -> List[float]:
        tensor = processor(image).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            return emb.flatten().tolist()
