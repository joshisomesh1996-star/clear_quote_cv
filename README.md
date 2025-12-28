# Car Orientation Classification – ClearQuote Assignment

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Labeling Logic](#labeling-logic)
4. [Data Augmentation](#data-augmentation)
5. [Model Used](#model-used)
6. [Training Parameters and Hyperparameters](#training-parameters-and-hyperparameters)
7. [Inference Pipeline](#inference-pipeline)
8. [Instructions to Run on a New System](#instructions-to-run-on-a-new-system)

---

## Problem Overview

The objective of this task is to build a real-time, edge-compatible computer vision solution that classifies the orientation of a car from an input image. The model identifies whether an image belongs to one of the following categories:

- front  
- frontleft  
- frontright  
- rear  
- rearleft  
- rearright  

Images that do not belong to any of these categories are classified as **random**.  
The final solution is converted to **TensorFlow Lite (TFLite)** format, and an inference pipeline is provided to run predictions on a folder of test images.

---

## Dataset Preparation

The dataset consists of multiple folders containing car images. Images are divided into folders only for readability; all images are treated uniformly during training.

Annotations are provided in **VIA (VGG Image Annotator) JSON format**, stored folder-wise. Each image contains multiple annotated car parts under the `identity` attribute. The actual image filename is taken from the `filename` field in the JSON annotation, not from the VIA-generated key.

For each image, all annotated part identities are extracted and used to infer the overall vehicle orientation. Annotation classes that are not relevant to orientation determination are ignored.

---

## Labeling Logic

A rule-based approach was used to assign orientation labels from annotations.

Front orientation was inferred using parts such as bonnet, front bumper, windshield, headlamp, and their partial variants. Rear orientation was inferred using parts such as rear bumper, tailgate, rear windshield, taillamp, and their partial variants.

Left and right orientation was inferred using annotated part names prefixed with **left** or **right**.

The final label was assigned by combining:
- face (front or rear)
- side (left, right, or none)

Images where orientation could not be reliably inferred were labeled as **random** to avoid introducing noise into the primary classes.

---

## Data Augmentation

During training, light data augmentation was applied in the form of **random brightness and contrast adjustments** to improve generalization. No geometric transformations were applied to preserve orientation semantics.

No augmentation was applied during validation or inference.

---

## Model Used

The model architecture is based on **MobileNetV2**, selected for its efficiency and suitability for edge deployment. The backbone was initialized with **ImageNet pretrained weights** and fine-tuned for the car orientation classification task.

---

## Training Parameters and Hyperparameters

The model was trained using **categorical cross-entropy loss** and the **Adam optimizer**. Transfer learning was applied by freezing the majority of the backbone layers and selectively unfreezing the final inverted residual blocks.

Key training and tuning parameters included:
- learning rate
- weight decay
- dropout rate
- batch size
- number of unfrozen MobileNetV2 blocks

Hyperparameter tuning was performed using an automated search strategy, and the configuration achieving the best validation performance was selected for final training. Early stopping based on validation loss was used to prevent overfitting.

---

## Inference Pipeline

After training, the best-performing model was converted to **TensorFlow Lite (TFLite)** format. The inference pipeline loads the TFLite model, preprocesses input images, performs prediction, and stores results in a **pandas DataFrame**.

For each image in the test folder, the pipeline records:
- image name
- predicted orientation

The DataFrame is saved as a **CSV file** for evaluation.

---

## Instructions to Run on a New System

The project can be executed on any system with **Python version 3.10 or above**.

### Step 1: Clone the repository and open it in your local environment

```bash
git clone https://github.com/joshisomesh1996-star/clear_quote_cv.git
cd clear_quote_cv
```

### Step 2: (Optional) Create and Activate Conda Environment

```bash
conda create -n cq_cv python=3.10 -y
conda activate cq_cv
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Inference on Test Images

```bash
python test_predict.py --test_dir <path_to_test_images>
```
