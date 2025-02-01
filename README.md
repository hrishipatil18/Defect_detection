# Defect Detection Pipeline
Defect Detection Pipeline
This project implements an end-to-end pipeline for detecting defects in manufacturing components using computer vision and deep learning techniques. The pipeline consists of several steps including data preprocessing, model training, evaluation, monitoring, and an automated training pipeline.
# Overview
The goal of this project is to build a defect detection system that can classify images of manufacturing components as either defective or non-defective. The pipeline utilizes deep learning models to automatically detect defects based on input images. The system is built in a modular way, allowing for easy scaling, modifications, and improvements.

# Project Structure

    defect_detection/
    │── dataset/                  # Downloaded dataset
    │── models/                   # Saved trained models
    │── notebooks/
    │   ├── defect_detection.ipynb # Main notebook for training and evaluation
    │── scripts/
    │   ├── preprocess.py          # Data preprocessing script
    │   ├── train.py               # Model training script
    │   ├── evaluate.py            # Model evaluation script
    │   ├── monitor.py             # Monitoring script (MLflow)
    │   ├── train_pipeline.py      # Python script to automate the entire training pipeline.
    │── requirements.txt           # Dependencies
    │── README.md                  # Project overview and instructions

# Important Files:
defect_detection.ipynb: Main notebook for model training and evaluation.
test.ipynb: Testing notebook for evaluating the trained model on unseen data.
train.py: Python script for training the model. Can be used for automation.
test.py: Python script for testing the model. Can be used for batch inference.
train_pipeline.py: Python script to automate the entire training pipeline.
dataset/: Folder containing the image dataset for training.

# Using run model training automated pipeline loacally

1. Clone the Repository
Start by cloning the repository to your local machine:

        bash
        
        git clone https://github.com/hrishipatil18/Defect_detection.git
        cd defect_detection_pipeline
or 
    download from google drive: https://drive.google.com/drive/folders/1aEgboB6IBxV-zbTJPEfeEZrnwQz3AamZ?usp=sharing
    
2. Install Dependencies
Install the required Python dependencies by running:

        bash
        pip install -r requirements.txt
3. Run training pipeline :

    bash 

    python scripts/train_pipeline.py

# Using Google colab

google drive: https://drive.google.com/drive/folders/1aEgboB6IBxV-zbTJPEfeEZrnwQz3AamZ?usp=sharing

Ensure the folder is copied into your Google Drive (e.g., MyDrive/defect_detection) this might have been in shared with me in your system.

Use Colab notebook to run pipeline:

https://colab.research.google.com/drive/10q-mA6IqbqlZ-9wiqa95Jki0dLB-UNXG?usp=sharing

# Model Evaluation
After training the model, evaluation metrics such as accuracy, precision, recall, and F1-score will be printed to assess performance.

Metrics are calculated using:

Accuracy: Overall classification accuracy.

Precision: True positive rate for defect class.

Recall: Sensitivity for detecting defects.

F1-Score: Harmonic mean of precision and recall.

# Automated Training Pipeline
The Automated Training Pipeline automates the entire process of data preprocessing, model training, evaluation, and monitoring. This script allows you to quickly run the whole training process with a single command.

This will:

Preprocess the data.

Train the model.

Save the trained model to the models/ directory.

Evaluate the model on the test dataset.

Monitor the training process.

# Notes
The dataset should be stored in the dataset/ directory.

The training script saves the model to the models/ directory as name_of_model.h5.

You can modify the model architecture, data preprocessing steps, or any other part of the pipeline to suit your specific needs.

The training, evaluation, and monitoring scripts are designed to be modular and easy to extend.
