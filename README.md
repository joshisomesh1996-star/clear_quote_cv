# Instructions to Run the Project

## 1. Create and Activate Conda Environment

Create the conda environment and activate it using the commands below:

conda create -n cq_cv python=3.10 -y  
conda activate cq_cv  

---

## 2. Install Dependencies

From the project root directory, install the required dependencies:

pip install -r requirements.txt  

---

## 3. Project Structure

The following project structure must be preserved:

project_root/  
├── test_predict.py  
├── readme.txt  
├── requirements.txt  
├── models/  
│   └── orientation_mobilenet_v2.tflite  
├── src/  
│   ├── config.py  
│   ├── preprocessing.py  
│   ├── tflite_infer.py  
│   └── pipeline.py  

---

## 4. Prepare Test Images

Place all test images directly inside a single folder (no subfolders):

test_images/  
├── img1.jpg  
├── img2.png  
├── img3.jpeg  

Supported image formats:
- jpg
- jpeg
- png
- webp

---

## 5. Run Inference

From the project root directory, run the following command:

python test_predict.py --test_dir test_images  

---

## 6. Specify Custom Output File (Optional)

To specify a custom output file name, run:

python test_predict.py --test_dir test_images --output predictions.csv  

---

## 7. Output

The output will be a CSV file named `predictions.csv` containing the following columns:
- image_name
- prediction

---

## 8. Notes

- The pipeline runs entirely on CPU
- GPU or CUDA is not required
- TensorFlow startup warnings can be safely ignored
- Corrupt or unreadable images are automatically labeled as `random`
