# Car Orientation Classification Pipeline – ClearQuote Assignment

This project implements an end-to-end computer vision pipeline to classify the orientation of a car from an image. The pipeline predicts one of the following orientation classes:

- front
- frontleft
- frontright
- rear
- rearleft
- rearright
- random

The solution is designed for edge deployment and uses a TensorFlow Lite (TFLite) model for CPU-only inference.

# Car Orientation Classification Pipeline – ClearQuote Assignment

This project implements an end-to-end computer vision pipeline to classify the orientation of a car from an image. The pipeline predicts one of the following orientation classes:

- front
- frontleft
- frontright
- rear
- rearleft
- rearright
- random

The solution is designed for edge deployment and uses a TensorFlow Lite (TFLite) model for CPU-only inference.

## Pipeline Description

The pipeline performs the following steps:

1. Parses annotated car-part information from VIA JSON files.
2. Applies a rule-based labeling strategy to assign orientation labels.
3. Trains a deep learning model to learn orientation patterns.
4. Converts the trained model to TensorFlow Lite format.
5. Runs inference on a folder of test images and stores predictions in a pandas DataFrame.

The inference pipeline is modular and can be executed using a single command.

## Labeling Flowchart

The rule-based labeling logic described above is illustrated in the following flowchart.

*(Insert labeling flowchart image here: label_logic_flow_chart.png)*

## Model Training

The model was trained using MobileNetV2 as the backbone architecture, initialized with ImageNet pretrained weights. This choice provides a good balance between accuracy and computational efficiency.

All images were resized to 224×224, converted to RGB format, and normalized using ImageNet mean and standard deviation values.

Transfer learning was applied by freezing the majority of the backbone layers initially and selectively unfreezing the last few inverted residual blocks to allow fine-tuning on the car orientation dataset.

The model was trained using categorical cross-entropy loss and the Adam optimizer. Early stopping was applied based on validation loss to prevent overfitting. The best-performing model checkpoint was saved and later converted to TensorFlow Lite format for deployment.

## Hyperparameter Tuning

Hyperparameter tuning was performed using an automated search strategy to improve validation performance.

The following parameters were tuned:
- Learning rate
- Weight decay
- Dropout rate
- Number of unfrozen MobileNetV2 blocks
- Batch size

Each trial trained the model for a limited number of epochs and evaluated validation accuracy. Early pruning was applied to stop underperforming trials. The configuration that achieved the highest validation accuracy was selected for final training.

This tuning process helped balance generalization performance and model stability while keeping the architecture lightweight for edge deployment.

## Training Flow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Preprocess Images]
    B --> C[Split Train and Validation Sets]
    C --> D[Load Pretrained MobileNetV2]
    D --> E[Freeze Base Layers]
    E --> F[Unfreeze Selected Blocks]
    F --> G[Train Model]
    G --> H[Validate Model]
    H --> I{Early Stopping}
    I -- No --> G
    I -- Yes --> J[Save Best Model]
    J --> K[Convert Model to TFLite]


---

## 🔹 BLOCK 8 — Inference Pipeline

```markdown
## Inference Pipeline

The inference pipeline loads the TensorFlow Lite model, preprocesses input images, performs prediction, and aggregates results.

All predictions are stored in a pandas DataFrame with two columns: image_name and prediction. The DataFrame is saved as a CSV file for easy evaluation and analysis.

## Instructions to Run on a New System

Create and activate a conda environment (optional but recommended):

conda create -n cq_cv python=3.10 -y  
conda activate cq_cv  

Alternatively, the project can be run using any Python version 3.10 or above.

Install dependencies from the project root directory:

pip install -r requirements.txt  

Prepare test images by placing all images directly inside a single folder without subfolders.

Run inference using the following command:

python test_predict.py --test_dir test_images  

To specify a custom output file name:

python test_predict.py --test_dir test_images --output predictions.csv  

## Notes

- The pipeline runs entirely on CPU.
- GPU or CUDA is not required.
- TensorFlow startup warnings can be safely ignored.
- Corrupt or unreadable images are automatically labeled as `random`.

## Conclusion

This project demonstrates a complete and production-ready computer vision pipeline, combining rule-based dataset labeling, deep learning model training, and edge-compatible inference with a clean and reproducible design.
