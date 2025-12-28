# src/preprocessing.py

import numpy as np
from PIL import Image

from src.config import IMG_SIZE

# Normalization values (ImageNet – same as training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads an image, preprocesses it, and returns a tensor
    ready for TFLite inference.

    Output shape: (1, IMG_SIZE, IMG_SIZE, 3)
    """

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy
    img = np.asarray(img).astype(np.float32) / 255.0

    # Normalize
    img = (img - MEAN) / STD

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
