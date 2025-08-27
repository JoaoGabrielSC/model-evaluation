from utils.file import prepare_dataset_directories
from config import DEST_BASE_DIR, UNKNOWN_RATIO
from utils.file import collect_image_data, split_known_and_unknown
from process.copier import process_known_persons, process_unknown_persons


def main():
    prepare_dataset_directories(DEST_BASE_DIR)

    person_counts, image_paths = collect_image_data()
    known, unknown = split_known_and_unknown(person_counts, UNKNOWN_RATIO)

    process_known_persons(known, image_paths)
    process_unknown_persons(unknown, image_paths)


if __name__ == "__main__":
    main()
