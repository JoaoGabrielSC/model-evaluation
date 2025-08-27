import os
import re


def get_jpg_files(directory: str) -> list[str]:
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".jpg") and re.match(r".+_\d+\.jpg", f)
    ]
