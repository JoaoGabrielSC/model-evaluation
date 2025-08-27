import math
import os
from sys import stdout
import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score
from repositories.embedding import FaceEmbeddingRepository
from factory.embedding_factory import EmbeddingFactory
from strategies.strategy_factory import get_strategy
from utils.normalize_image import normalize_image
from utils.extract_face import extract_face
from data.image_loader import get_jpg_files
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.make_mosaic import save_mosaic

METRIC_OPS = {"cosine": "<=>", "euclidean": "<->", "dot_product": "<#>"}
THRESHOLDS = {
    "<=>": 0.1,
    "<->": 10.0,
    "<#>": 0.1,
}


class FaceEmbeddingValidator:
    def __init__(self, model_name: str, folder_path: str, now: str):
        self.model_name = model_name
        self.folder_path = folder_path
        self.embedding = EmbeddingFactory.get(model_name)
        self.strategy = get_strategy(model_name)
        self.embedding.model.eval()
        self.result_dir = os.path.join("results", now, model_name)
        os.makedirs(self.result_dir, exist_ok=True)

    def validate(self, dist_type: str):
        return self._validate_distance(dist_type)

    def _validate_distance(self, dist_type: str):
        metric_op = METRIC_OPS[dist_type]
        table_name = f"face_embeddings_{self.model_name}"
        match_images = []
        mismatch_images = []
        unknown_images = []
        y_true = []
        y_pred = []

        log_path = os.path.join(
            self.result_dir, f"log_{self.model_name}_{dist_type}.txt"
        )
        with open(log_path, "w") as log_file, FaceEmbeddingRepository() as repo:
            people_dirs = self._get_people_dirs(self.folder_path)

            for person in tqdm(
                people_dirs,
                desc=f"{self.model_name} ({dist_type})",
                unit="jpg file",
                file=stdout,
            ):
                person_dir = os.path.join(self.folder_path, person)
                all_files = get_jpg_files(person_dir)
                half = len(all_files) // 3
                for file_path in all_files[:half]:

                    filename = os.path.basename(file_path)
                    face_image = extract_face(file_path)
                    if face_image is None:
                        log_file.write(f"[{person}] {filename} => FACE NÃO DETECTADA\n")
                        continue

                    normalized = normalize_image(face_image)
                    embedding_vector = self.strategy.get_embedding(
                        self.embedding.model, self.embedding.processor, normalized
                    )
                    threshold = THRESHOLDS[metric_op]
                    results = repo.search_similar(
                        table_name,
                        embedding_vector,
                        metric_op,
                        limit=1,
                        threshold=threshold,
                    )

                    pred_name, score = results[0] if results else ("unknown", -1)
                    annotated = self._annotate_image(
                        normalized, person, pred_name, score
                    )

                    y_true.append(person)
                    y_pred.append(pred_name)

                    if results:
                        if pred_name == person:
                            match_images.append(annotated)
                            log_file.write(
                                f"[{person}] {filename} => CORRETO: {pred_name}, SCORE: {score:.2f}\n"
                            )
                        else:
                            mismatch_images.append(annotated)
                            log_file.write(
                                f"[{person}] {filename} => ERRADO: {pred_name}, SCORE: {score:.2f}\n"
                            )
                    else:
                        unknown_images.append(annotated)
                        log_file.write(
                            f"[{person}] {filename} {score} => NENHUM RESULTADO\n"
                        )

        dist_dir = os.path.join(self.result_dir, dist_type)
        os.makedirs(dist_dir, exist_ok=True)

        save_mosaic(dist_dir, match_images, "match")
        save_mosaic(dist_dir, mismatch_images, "mismatch")
        save_mosaic(dist_dir, unknown_images, "unknown")

        if y_true and y_pred:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )

            labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

            fig_width = max(8, len(labels) * 0.7)
            fig_height = max(6, len(labels) * 0.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            disp.plot(
                include_values=True,
                xticks_rotation=45,
                cmap="Blues",
                colorbar=False,
                ax=ax,
            )

            plt.title(f"Matriz de Confusão - {self.model_name} [{dist_type}]")
            plt.tight_layout()

            cm_path = os.path.join(dist_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()

            metrics_txt = (
                f"\n=== MÉTRICAS PARA {self.model_name.upper()} [{dist_type}] ===\n"
                f"Accuracy:  {acc:.4f}\n"
                f"F1-Score:  {f1:.4f}\n"
                f"Precision: {precision:.4f}\n"
            )
            with open(log_path, "a") as log_file:
                log_file.write(metrics_txt)
            print(metrics_txt)

        return (
            {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
            }
            if y_true and y_pred
            else None
        )

    def _get_people_dirs(self, base_path: str) -> List[str]:
        return [
            name
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ]

    def _annotate_image(
        self, image: Image.Image, expected: str, predicted: str, score: float
    ) -> np.ndarray:
        np_img = np.array(image.convert("RGB"))[:, :, ::-1]

        if np_img.max() <= 1.0:
            np_img = (np_img * 255).astype(np.uint8)

        img = cv2.resize(np_img, (100, 100))
        annotated = cv2.copyMakeBorder(
            img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        expected_first = expected.split()[0] if expected else ""
        predicted_first = predicted.split()[0] if predicted else ""

        lines = [f"E: {expected_first}", f"P: {predicted_first}", f"S: {score:.2f}"]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        line_height = 15
        x, y_start = 5, 20

        overlay = annotated.copy()
        alpha = 0.6

        box_height = line_height * len(lines) + 5
        box_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            box_width = max(box_width, w)

        cv2.rectangle(
            overlay,
            (x - 3, y_start - 15),
            (x + box_width + 3, y_start + box_height - 5),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        for i, line in enumerate(lines):
            y = y_start + i * line_height
            cv2.putText(annotated, line, (x, y), font, font_scale, color, thickness)

        return annotated
