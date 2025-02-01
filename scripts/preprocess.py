import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

# Load dataset from Kaggle or another source
def load_dataset(dataset_path):
    image_paths = []
    labels = []
    
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(label)
    
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

# Split dataset
def split_dataset(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    return train_df, test_df

# Data Augmentation
def create_data_generators(train_df, test_df, img_size=(227, 227), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                 height_shift_range=0.2, horizontal_flip=True)
    
    train_generator = datagen.flow_from_dataframe(
        train_df, x_col='image_path', y_col='label', target_size=img_size,
        batch_size=batch_size, class_mode='categorical' if len(train_df['label'].unique()) == 2 else 'categorical'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        test_df, x_col='image_path', y_col='label', target_size=img_size,
        batch_size=batch_size, class_mode='categorical' if len(train_df['label'].unique()) == 2 else 'categorical'
    )

    return train_generator, test_generator
