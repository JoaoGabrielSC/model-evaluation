import os
import shutil
import random
import os
from collections import defaultdict
from typing import Dict, List

from config import SOURCE_CROPPED_DIR


def prepare_dataset_directories(dest_base: str) -> None:
    dirs = ["embedding_faces", "test", "val"]
    for d in dirs:
        path = os.path.join(dest_base, d)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


def collect_image_data() -> tuple[Dict[str, int], Dict[str, List[str]]]:
    person_image_count: dict[str, int] = defaultdict(int)
    image_paths = defaultdict(list)

    for filename in os.listdir(SOURCE_CROPPED_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        person = filename.rsplit("_", 1)[0]
        person_image_count[person] += 1
        image_paths[person].append(filename)

    return person_image_count, image_paths


def split_known_and_unknown(
    person_image_count: Dict[str, int], ratio: float
) -> tuple[set, set]:
    all_persons = list(person_image_count.keys())
    unknown_count = int(len(all_persons) * ratio)
    unknown = set(random.sample(all_persons, unknown_count))
    known = set(all_persons) - unknown
    return known, unknown
