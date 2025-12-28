# src/tflite_infer.py

import tensorflow as tf
import numpy as np

from src.config import MODEL_PATH


class TFLiteModel:
    def __init__(self):
        # Load TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input & output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cache indices for speed
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

    def predict(self, input_tensor: np.ndarray) -> int:
        """
        Runs inference on a single preprocessed image tensor.

        input_tensor shape: (1, H, W, 3)
        returns: predicted class index (int)
        """

        # Set input tensor
        self.interpreter.set_tensor(self.input_index, input_tensor)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_index)

        # Argmax over classes
        pred_idx = int(np.argmax(output[0]))

        return pred_idx
