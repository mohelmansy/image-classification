# Image Classification with EfficientNet and ResNet

## Overview

This project is an implementation of an image classification model using **EfficientNetB0** and **ResNet50** architectures. It employs **TensorFlow** and **Keras** for model training and evaluation. The models are trained on an augmented dataset with binary classification (Fake vs Real).

## Features

- Uses **EfficientNetB0** and **ResNet50** pretrained models
- Implements **image augmentation** for better generalization
- Utilizes **Early Stopping**, **Learning Rate Reduction**, and **Model Checkpointing**
- Applies **fine-tuning** after initial training to improve model accuracy
- Saves trained models and history for further analysis
- Includes visualization functions to plot accuracy trends
- Provides a function for **real-time image prediction**

## Dataset Preparation

Place your dataset in a folder named `dataset`, structured as follows:

```
dataset/
    ├── class_1/  # Images belonging to class 1
    ├── class_2/  # Images belonging to class 2
```

Update `dataset_path` in the script if your dataset location differs.

## Dependencies

Ensure you have the following libraries installed:

```bash
pip install tensorflow numpy matplotlib seaborn
```

## Code Structure

- **Data Augmentation**: Extensive transformations applied to training images.
- **Model Building**: Two models (EfficientNet & ResNet) built dynamically with custom dense layers.
- **Training & Fine-Tuning**: Models are first trained, then fine-tuned for better performance.
- **Visualization**: Accuracy trends plotted for both training and validation.
- **Prediction Function**: Allows real-time image classification.

## Training Process

1. Load the dataset using `ImageDataGenerator`
2. Train EfficientNet and ResNet models
3. Fine-tune both models
4. Plot accuracy graphs
5. Save the trained models

## Model Training & Fine-Tuning

```python
# Train EfficientNet model
efficientnet_model = build_model(EfficientNetB0, "efficientnet")
history_efficientnet = train_model(efficientnet_model, train_data, val_data, "efficientnet")

# Train ResNet model
resnet_model = build_model(ResNet50, "resnet")
history_resnet = train_model(resnet_model, train_data, val_data, "resnet")

# Fine-tune models
history_efficientnet_fine = fine_tune_model(efficientnet_model, train_data, val_data, "efficientnet")
history_resnet_fine = fine_tune_model(resnet_model, train_data, val_data, "resnet")
```

## Model Evaluation

The training history is stored and can be plotted using:

```python
plot_history(
    [history_efficientnet, history_resnet],
    ["EfficientNet Training", "ResNet Training"]
)
```

## Predicting a New Image

To classify a new image, use:

```python
result = predict_image(model, "path_to_image.jpg")
print(result)  # Output: "Fake" or "Real"
```

## Checkpoints & Model Saving

- The best model is saved automatically as `{model_name}_best.h5`
- The trained model object is pickled for later use (`.pkl` file)

## Contributors

Feel free to contribute to this project by submitting pull requests!

## License

This project is licensed under the MIT License.

