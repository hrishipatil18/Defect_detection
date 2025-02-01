import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16 #MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from preprocess import load_dataset, split_dataset, create_data_generators

# Load and preprocess data
dataset_path = "./dataset"
df = load_dataset(dataset_path)
train_df, test_df = split_dataset(df)
train_generator, test_generator = create_data_generators(train_df, test_df)

# Model Architectures
def build_model(base_model_name="ResNet50", input_shape=(227, 227, 3), num_classes=2):
    if base_model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    #elif base_model_name == "MobileNetV2":
    #    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    else:
        raise ValueError("Invalid model name")

    base_model.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="sigmoid" if num_classes == 2 else "softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="binary_crossentropy" if num_classes == 2 else "categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model 

# Train and save models
resnet_model = build_model("ResNet50", num_classes=len(df['label'].unique()))
#mobilenet_model = build_model("MobileNetV2", num_classes=len(df['label'].unique()))
VGG16_model = build_model("VGG16", num_classes=len(df['label'].unique()))

resnet_model.fit(train_generator, validation_data=test_generator, epochs=15)
resnet_model.save("models/resnet_model.h5")

#mobilenet_model.fit(train_generator, validation_data=test_generator, epochs=2)
#mobilenet_model.save("models/mobilenet_model.h5")

VGG16_model.fit(train_generator, validation_data=test_generator, epochs=15)
VGG16_model.save("models/VGG16_model.h5")