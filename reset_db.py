from repositories.embedding import FaceEmbeddingRepository
from enums import ModelType


def reset_face_embeddings_table():
    with FaceEmbeddingRepository() as embedding_repository:
        models = list(ModelType)
        for model_name in models:
            embedding_repository.delete_table(model_name.value)
    print("✅ Banco reiniciado com sucesso.")


if __name__ == "__main__":
    reset_face_embeddings_table()
