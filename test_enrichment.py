import os
from process.face_data_enrichment import FaceDataEnricher
from utils.extract_face import extract_face
from utils.make_mosaic import build_mosaic, save_mosaic
import cv2

image_path = "datasets/embedding_faces/Akshay Kumar/Akshay Kumar_0.jpg"
original_face = extract_face(image_path)

if original_face is None:
    raise ValueError("No face found in the image.")

enricher = FaceDataEnricher(num_augmented=10)
augmented_faces = enricher.enrich(original_face)

output_dir = "datasets/augmented_faces"
os.makedirs(output_dir, exist_ok=True)

if not augmented_faces:
    raise ValueError("No augmented faces generated.")

print(f"Generated {len(augmented_faces)} augmented faces.")

save_mosaic(output_dir, augmented_faces, "augmented")
