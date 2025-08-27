import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from enums import ModelType
from process.face_embedding_builder import FaceEmbeddingBuilder
from utils.certificate import configure_ssl_certifi


def get_model_names(selected: str | None) -> list[str]:
    if selected:
        return [selected.lower()]
    return [m.value for m in ModelType]


def build_embeddings_for_model(model_name: str):
    print("#" * 40)
    print(f"🔄 Processing model: {model_name}")
    try:
        builder = FaceEmbeddingBuilder(model_name)
        builder.generate_embeddings()
    except Exception as e:
        print(f"❌ Error processing model '{model_name}': {e}")


def main():
    configure_ssl_certifi()

    parser = argparse.ArgumentParser(
        description="Populate pgvector table with embeddings from images"
    )
    parser.add_argument("--model", type=str, help="Model type")
    args = parser.parse_args()

    model_names = get_model_names(args.model)

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(build_embeddings_for_model, name) for name in model_names
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
