# Defect Detection Pipeline
Defect Detection Pipeline
This project implements an end-to-end pipeline for detecting defects in manufacturing components using computer vision and deep learning techniques. The pipeline consists of several steps including data preprocessing, model training, evaluation, monitoring, and an automated training pipeline.

Overview
The goal of this project is to build a defect detection system that can classify images of manufacturing components as either defective or non-defective. The pipeline utilizes deep learning models to automatically detect defects based on input images. The system is built in a modular way, allowing for easy scaling, modifications, and improvements.

#Project Structure

    defect_detection/
    │── dataset/                  # Downloaded dataset
    │── models/                   # Saved trained models
    │── notebooks/
    │   ├── defect_detection.ipynb # Main notebook for training and evaluation
    │── scripts/
    │   ├── preprocess.py          # Data preprocessing script
    │   ├── train.py               # Model training script
    │   ├── evaluate.py            # Model evaluation script
    │   ├── monitor.py             # Monitoring script (MLflow/TensorBoard)
    │── requirements.txt           # Dependencies
    │── README.md                  # Project overview and instructions
    Installation
