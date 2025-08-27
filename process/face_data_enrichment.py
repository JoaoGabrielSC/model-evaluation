import cv2
import numpy as np
import random
from typing import List, Union
from PIL import Image


class FaceDataEnricher:
    def __init__(self, num_augmented: int = 5):
        self.num_augmented = num_augmented
        self.transformations = [
            self.__add_shadow,
            self.__random_crop,
            self.__adjust_brightness,
            self.__horizontal_flip,
            self.__rotate_image,
        ]

    def enrich(self, image: Union[np.ndarray, Image.Image]) -> List[np.ndarray]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        enriched_images = []
        for _ in range(self.num_augmented):
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = self.__apply_random_transformations(img)
            enriched_images.append(img)
        return enriched_images

    def __apply_random_transformations(self, image: np.ndarray) -> np.ndarray:

        random.shuffle(self.transformations)
        for t in self.transformations[: random.randint(1, 3)]:
            image = t(image)
        return image

    def __rotate_image(self, image: np.ndarray) -> np.ndarray:
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def __horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            return cv2.flip(image, 1)
        return image

    def __adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        factor = random.uniform(0.6, 1.4)
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def __random_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_crop = int(min(h, w) * 0.1)
        top = random.randint(0, max_crop)
        left = random.randint(0, max_crop)
        bottom = h - random.randint(0, max_crop)
        right = w - random.randint(0, max_crop)
        return cv2.resize(image[top:bottom, left:right], (w, h))

    def __add_shadow(self, image: np.ndarray) -> np.ndarray:
        top_x, bottom_x = random.randint(0, image.shape[1]), random.randint(
            0, image.shape[1]
        )
        shadow_mask = np.zeros_like(image[:, :, 0])
        X_m, Y_m = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        mask = (X_m - 0) * (bottom_x - top_x) - (image.shape[0] - 0) * (
            Y_m - top_x
        ) >= 0
        shadow_mask[mask] = 1
        shadow_intensity = random.uniform(0.4, 0.75)
        for i in range(3):
            image[:, :, i] = np.where(
                shadow_mask == 1, image[:, :, i] * shadow_intensity, image[:, :, i]
            )
        return image
