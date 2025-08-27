import argparse
from datetime import datetime
from utils.plot import plot_metric_histograms
from validator.embedding_validator import FaceEmbeddingValidator
import os
from multiprocessing import Pool

ALL_MODELS = [
    "clip",
    "vit",
    "deit",
    "dino",
    "resnet50",
    "resnet18",
    "effnet_b0",
    "mobilenet",
]
DISTANCES = ["cosine", "euclidean", "dot_product"]


def get_model_names(selected: str | None) -> list[str]:
    if selected:
        return [selected.lower()]
    return ALL_MODELS


def validate_model_distance(args):
    model_name, dist_type, now = args
    validator = FaceEmbeddingValidator(model_name, "datasets/val", now=now)
    metrics = validator.validate(dist_type)
    if metrics:
        print(f"Modelo: {model_name}, Distância: {dist_type}, " f"Métricas: {metrics}")
        return {"model": model_name, "dist": dist_type, **metrics}
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate embeddings from pgvector database"
    )
    parser.add_argument("--model", type=str, help="Model name to validate (optional)")
    args = parser.parse_args()
    now = datetime.now().strftime("%d-%m_%H:%M:%S")

    model_names = get_model_names(args.model)

    tasks = [
        (model_name, dist_type, now)
        for model_name in model_names
        for dist_type in DISTANCES
    ]

    with Pool(processes=2) as pool:
        results = pool.map(validate_model_distance, tasks)

    all_results = [r for r in results if r]

    if all_results:
        out_dir = os.path.join("results", now, "summary")
        os.makedirs(out_dir, exist_ok=True)
        plot_metric_histograms(all_results, out_dir)


if __name__ == "__main__":
    main()
