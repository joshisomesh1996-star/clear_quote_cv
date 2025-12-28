# src/pipeline.py

import os
from typing import List, Tuple

from src.preprocessing import preprocess_image
from src.tflite_infer import TFLiteModel
from src.config import CLASS_NAMES, IMAGE_EXTENSIONS


def run_inference_on_folder(test_dir: str) -> List[Tuple[str, str]]:
    """
    Runs inference on all images inside test_dir.

    Returns:
        List of (image_name, predicted_class)
    """

    # Initialize model once
    model = TFLiteModel()

    results = []

    # Sort for deterministic output
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)

        try:
            # Preprocess image
            input_tensor = preprocess_image(img_path)

            # Predict
            pred_idx = model.predict(input_tensor)
            pred_label = CLASS_NAMES[pred_idx]

            results.append((img_name, pred_label))

        except Exception as e:
            # If any image fails, mark as random
            results.append((img_name, "random"))

    return results
