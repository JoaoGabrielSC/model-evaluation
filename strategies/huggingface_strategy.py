from .embedding_strategy import EmbeddingStrategy
import numpy as np
import torch
from typing import List


class HuggingFaceStrategy(EmbeddingStrategy):
    def get_embedding(self, model, processor, image) -> List[float]:
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            _embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()[0]
            embedding = _embedding / np.linalg.norm(_embedding)
            flatted_emb = embedding.flatten().tolist()
            return flatted_emb
