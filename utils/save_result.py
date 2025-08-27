import os
import cv2



def save_result(
    model_name: str,
    distance: str,
    image_path: str,
    person: str,
    pred_name: str,
    score: float,
) -> None:
    save_dir = f"./false_positive/{model_name}/{distance}"
    os.makedirs(save_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is not None:
        cv2.putText(
            image_bgr,
            f"Expected: {person}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            f"Predicted: {pred_name}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            f"Score: {score:.4f}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(save_dir, filename), image_bgr)
