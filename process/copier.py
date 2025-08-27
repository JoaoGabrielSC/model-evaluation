import os
import shutil
from typing import List
from tqdm import tqdm

from config import DEST_BASE_DIR, SOURCE_ORIGINAL_DIR, SPLITS


def copy_images(
    image_list: List[str],
    person: str,
    start_index: int,
    end_index: int,
    split_name: str,
):
    for image_name in image_list[start_index:end_index]:
        src_path = os.path.join(SOURCE_ORIGINAL_DIR, person, image_name)
        dest_dir = os.path.join(DEST_BASE_DIR, split_name, person)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_path, dest_dir)


def process_known_persons(known_persons: set, image_paths: dict[str, List[str]]):
    for person in tqdm(known_persons, desc="Processing known persons"):
        images = sorted(image_paths[person])
        total_images = len(images)
        emb_count = int(SPLITS["embedding_faces"] * total_images)
        test_count = int(SPLITS["test"] * total_images)

        copy_images(images, person, 0, emb_count, "embedding_faces")
        copy_images(images, person, emb_count, emb_count + test_count, "test")


def process_unknown_persons(unknown_persons: set, image_paths: dict[str, List[str]]):
    for person in tqdm(unknown_persons, desc="Processing unknown persons (val only)"):
        for image_name in image_paths[person]:
            src_path = os.path.join(SOURCE_ORIGINAL_DIR, person, image_name)
            dest_dir = os.path.join(DEST_BASE_DIR, "val", person)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_path, dest_dir)
