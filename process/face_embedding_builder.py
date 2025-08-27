import os
import cv2
from typing import List
from repositories.embedding import FaceEmbeddingRepository
from factory.embedding_factory import EmbeddingFactory
from strategies.strategy_factory import get_strategy
from utils.normalize_image import normalize_image
from utils.extract_face import extract_face
from data.image_loader import get_jpg_files


SOURCE_DIR = "./datasets/embedding_faces"


class FaceEmbeddingBuilder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding = EmbeddingFactory.get(model_name)
        self.strategy = get_strategy(model_name)
        self.table_name = f"face_embeddings_{model_name}"
        self.vector_dimension = self.embedding.dim
        self.embedding.model.eval()

    def generate_embeddings(self):
        self._log_start()

        with FaceEmbeddingRepository() as repo:
            repo.create_table(self.table_name, self.vector_dimension)
            self._create_embeddings_for_all_people(repo)

        self._log_finish()

    def _create_embeddings_for_all_people(self, repo: FaceEmbeddingRepository):
        for person in self._get_people_dirs(SOURCE_DIR):
            person_dir = os.path.join(SOURCE_DIR, person)
            for file_path in get_jpg_files(person_dir):
                self._process_file(repo, file_path, person)

    def generate_embedding_from_pil_image(
        self, image: cv2.typing.MatLike, person: str
    ) -> List[float] | None:
        face_image = extract_face(image, from_array=True)
        if face_image is None:
            print("⚠️ No face found in provided image, skipping.")
            return None

        normalized = normalize_image(face_image)
        embedding = self.strategy.get_embedding(
            self.embedding.model, self.embedding.processor, normalized
        )

        with FaceEmbeddingRepository() as repo:
            repo.create_table(self.table_name, self.vector_dimension)
            self._save_embedding(repo, person, embedding)

        return embedding

    def _process_file(self, repo: FaceEmbeddingRepository, file_path: str, person: str):
        embedding = self._extract_embedding(file_path)
        if embedding is None:
            print(f"⚠️ No face found in {file_path}, skipping.")
            return

        self._save_embedding(repo, person, embedding)

    def _extract_embedding(self, file_path: str) -> List[float] | None:
        face_image = extract_face(file_path)
        if face_image is None:
            return None

        normalized = normalize_image(face_image)
        return self.strategy.get_embedding(
            self.embedding.model, self.embedding.processor, normalized
        )

    def _save_embedding(
        self, repo: FaceEmbeddingRepository, person: str, embedding: List[float]
    ):
        repo.insert_embedding(
            table_name=self.table_name,
            person=person,
            embedding=embedding,
            model_name=self.model_name,
        )

    def _get_people_dirs(self, base_path: str) -> List[str]:
        return [
            name
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ]

    def _log_start(self):
        print("=" * 40)
        print(f"📦 Processing model: {self.model_name}")

    def _log_finish(self):
        print(f"✅ Finished processing model: {self.model_name}")
