import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from preprocess import create_data_generators, split_dataset, load_dataset
import cv2

df = load_dataset("./dataset")
train_df, test_df = split_dataset(df)
_, test_generator = create_data_generators(train_df, test_df)

# Load trained models
resnet_model = tf.keras.models.load_model("models/resnet_model.h5")
#mobilenet_model = tf.keras.models.load_model("models/mobilenet_model.h5")
VGG16_model = tf.keras.models.load_model("models/VGG16_model.h5")
# Evaluate models
def evaluate_model(model, test_generator):
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)

    print(f"Model Evaluation:\n{classification_report(y_true, y_pred)}")

evaluate_model(resnet_model, test_generator)
#evaluate_model(mobilenet_model, test_generator)
evaluate_model(VGG16_model, test_generator)
