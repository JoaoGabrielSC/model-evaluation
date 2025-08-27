import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image
from typing import Union
import cv2
import mediapipe as mp

# 👇 Depois de importar o mediapipe
import absl.logging

absl.logging.set_verbosity("error")
absl.logging.set_stderrthreshold("error")


def extract_face(image_path: str) -> Union[Image.Image, None]:
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        result = detector.process(rgb)
        if not result.detections:
            return None

        box = result.detections[0].location_data.relative_bounding_box
        h, w, _ = rgb.shape
        x1 = max(int(box.xmin * w), 0)
        y1 = max(int(box.ymin * h), 0)
        x2 = min(x1 + int(box.width * w), w)
        y2 = min(y1 + int(box.height * h), h)
        face_crop = rgb[y1:y2, x1:x2]

    return Image.fromarray(face_crop)
