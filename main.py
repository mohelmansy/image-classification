import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import pickle
import os

# Enable mixed precision for performance boost
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define dataset path
dataset_path = "dataset"  # Update to actual dataset location

# Image augmentation with extensive transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.4, 1.6],
    fill_mode='nearest',
    validation_split=0.2
)

# Load dataset
def load_data(datagen, subset):
    return datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary',
        subset=subset,
        shuffle=True
    )

train_data = load_data(train_datagen, "training")
val_data = load_data(train_datagen, "validation")

# Define callbacks
def get_callbacks(model_name):
    return [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f"{model_name}_best.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]

# Function to build model dynamically
def build_model(base_model_class, model_name):
    base_model = base_model_class(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid", dtype=tf.float32)(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0002), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Train models
def train_model(model, train_data, val_data, model_name, epochs=35):
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=get_callbacks(model_name), verbose=1)
    with open(f"{model_name}_model.pkl", "wb") as file:
        pickle.dump(model, file)
    return history

# Build and train EfficientNet model
efficientnet_model = build_model(EfficientNetB0, "efficientnet")
history_efficientnet = train_model(efficientnet_model, train_data, val_data, "efficientnet")

# Build and train ResNet50 model
resnet_model = build_model(ResNet50, "resnet")
history_resnet = train_model(resnet_model, train_data, val_data, "resnet")

# Fine-tune models
def fine_tune_model(model, train_data, val_data, model_name, epochs=25):
    model.trainable = True
    model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=["accuracy"])
    return train_model(model, train_data, val_data, model_name + "_fine_tuned", epochs)

history_efficientnet_fine = fine_tune_model(efficientnet_model, train_data, val_data, "efficientnet")
history_resnet_fine = fine_tune_model(resnet_model, train_data, val_data, "resnet")

# Function to plot training history
def plot_history(histories, titles):
    for history, title in zip(histories, titles):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='-', marker='o', markersize=5)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='-', marker='s', markersize=5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.show()

plot_history(
    [history_efficientnet, history_resnet],
    ["EfficientNet Training", "ResNet Training"]
)

# Function to predict images
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

