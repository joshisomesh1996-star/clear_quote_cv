# Instructions to Run the Project

Create and activate the conda environment using the commands below.

conda create -n cq_cv python=3.10 -y
conda activate cq_cv

Install the required dependencies from the project root directory.

pip install -r requirements.txt

The project structure must be preserved as shown below.

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

Place all test images directly inside a single folder without subfolders.

test_images/
├── img1.jpg
├── img2.png
├── img3.jpeg

Supported image formats are .jpg, .jpeg, .png, and .webp.

Run inference from the project root using the following command.

python test_predict.py --test_dir test_images

To specify a custom output file name, use the command below.

python test_predict.py --test_dir test_images --output predictions.csv

The output will be a CSV file named predictions.csv containing two columns: image_name and prediction.

The pipeline runs entirely on CPU and does not require GPU or CUDA. TensorFlow startup warnings can be safely ignored. Corrupt or unreadable images are automatically labeled as random.
