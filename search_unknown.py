import os
import sys
import contextlib
from repositories.embedding import FaceEmbeddingRepository
from factory.embedding_factory import EmbeddingFactory
from strategies.strategy_factory import get_strategy
from utils.extract_face import extract_face
from utils.normalize_image import normalize_image


# IMAGE_PATH = "./datasets/val/Anushka Sharma/Anushka Sharma_0.jpg"
IMAGE_PATH = "./datasets/val//Marmik/Marmik_0.jpg"
MODEL_NAME = "clip"
DISTANCE_TYPE = "cosine"
METRIC_OPS = {"cosine": "<=>", "euclidean": "<->", "dot_product": "<#>"}


def convert_distance_to_score(distance, dist_type):
    if dist_type == "cosine":
        return 1 - (distance / 2)
    elif dist_type == "euclidean":
        max_dist = 2.0
        return max(0.0, 1 - (distance / max_dist))
    elif dist_type == "dot_product":
        return distance
    else:
        return 0


def main():
    embedding = EmbeddingFactory.get(MODEL_NAME)
    strategy = get_strategy(MODEL_NAME)
    embedding.model.eval()
    metric_op = METRIC_OPS[DISTANCE_TYPE]
    table_name = f"face_embeddings_{MODEL_NAME}"

    face = extract_face(IMAGE_PATH)
    if face is None:
        print(f"Nenhum rosto detectado na imagem: {IMAGE_PATH}")
        return

    normalized = normalize_image(face)
    vector = strategy.get_embedding(embedding.model, embedding.processor, normalized)

    with FaceEmbeddingRepository() as repo:
        results = repo.search_similar(
            table_name, vector, metric_op, limit=3, threshold=0.1
        )

    if results:
        print(f"\nResultados para imagem {os.path.basename(IMAGE_PATH)}:")
        for i, (name, dist) in enumerate(results, 1):
            score = convert_distance_to_score(dist, DISTANCE_TYPE)
            print(f"{i}. Nome: {name}, Score: {score:.4f}")
    else:
        print(f"Nenhum resultado encontrado. para {os.path.basename(IMAGE_PATH)}")


if __name__ == "__main__":
    main()
