from factory import EmbeddingFactory
from strategies.clip_strategy import ClipStrategy
import cv2

embedding = EmbeddingFactory.get("clip")

strategy = ClipStrategy()

image = cv2.imread("eu.jpeg")
embedding_vector = strategy.get_embedding(embedding.model, embedding.processor, image)

print(embedding_vector)
