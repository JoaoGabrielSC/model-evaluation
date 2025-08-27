from abc import ABC, abstractmethod
from PIL import Image
from typing import List


class EmbeddingStrategy(ABC):
    @abstractmethod
    def get_embedding(self, model, processor, image: Image.Image) -> List[float]:
        raise NotImplementedError(
            "Método 'get_embedding' deve ser implementado na subclasse."
        )
