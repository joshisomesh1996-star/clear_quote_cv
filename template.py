import os

# Folders to create in current directory
folders = [
    "models",
    "src",
]

# Files to create in current directory
files = [
    "test_predict.py",
    "readme.txt",
    "requirements.txt",

    "src/__init__.py",
    "src/config.py",
    "src/preprocessing.py",
    "src/tflite_infer.py",
    "src/pipeline.py",
]

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass
            print(f"Created file: {file}")
        else:
            print(f"Already exists: {file}")

if __name__ == "__main__":
    create_structure()
    print("\n✅ Project structure created in current directory")
