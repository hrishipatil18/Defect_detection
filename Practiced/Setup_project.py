import os

# Define the project structure
folders = [
    "dataset",
    "models",
    "notebooks",
    "scripts"
]

files = {
    "notebooks/defect_detection.ipynb": "",
    "scripts/preprocess.py": "",
    "scripts/train.py": "",
    "scripts/evaluate.py": "",
    "scripts/monitor.py": "",
    "scripts/train_pipeline.py": "",
    "requirements.txt": "tensorflow\nnumpy\npandas\nopencv-python\nscikit-learn\nmatplotlib\nmlflow",
    "README.md": "# Defect Detection Pipeline\n\n## Overview\nThis project implements a defect detection pipeline for manufacturing components using deep learning."
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("Project structure created successfully!")