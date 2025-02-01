import mlflow
import mlflow.tensorflow
import tensorflow as tf

mlflow.set_experiment("Defect Detection")

with mlflow.start_run():
    model = tf.keras.models.load_model("models/resnet_model.h5")
    mlflow.tensorflow.log_model(model, "resnet_model")
    mlflow.log_param("epochs", 10)
    mlflow.log_metric("accuracy", 0.92)  # Replace with actual accuracy
