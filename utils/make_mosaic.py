import os
import math
from typing import List
import cv2
import numpy as np


def save_mosaic(output_dir: str, images: List[np.ndarray], prefix: str):
    max_cols = 5
    max_rows = 5
    max_per_mosaic = max_cols * max_rows

    total_images = len(images)
    if total_images == 0:
        return

    for i in range(0, total_images, max_per_mosaic):
        chunk = images[i : i + max_per_mosaic]

        cur_cols = min(max_cols, math.ceil(math.sqrt(len(chunk))))
        cur_rows = min(max_rows, math.ceil(len(chunk) / cur_cols))

        while len(chunk) < cur_rows * cur_cols:
            chunk.append(np.zeros_like(chunk[0]))

        mosaic = build_mosaic(chunk, cur_rows, cur_cols)
        path = os.path.join(output_dir, f"{prefix}_{i // max_per_mosaic + 1:03d}.jpg")
        cv2.imwrite(path, mosaic)


def build_mosaic(images: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not images or cols == 0 or rows == 0:
        raise ValueError("Lista de imagens vazia ou grid inválido.")
    grid = []
    for r in range(rows):
        row = images[r * cols : (r + 1) * cols]
        if not row:  # Skip empty rows
            break
        # Pad row with blank images if needed
        while len(row) < cols:
            row.append(np.zeros_like(images[0]))
        grid.append(np.hstack(row))
    return np.vstack(grid)
