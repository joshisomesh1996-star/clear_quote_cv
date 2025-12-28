# src/config.py

IMG_SIZE = 224
NUM_CHANNELS = 3

# FINAL class order (from training checkpoint)
CLASS_NAMES = [
    "front",
    "frontleft",
    "frontright",
    "random",
    "rear",
    "rearleft",
    "rearright"
]

MODEL_PATH = "models/orientation_mobilenet_v2_compat.tflite"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
