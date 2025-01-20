# EasyOLO: A Simple YOLO Implementation for Object Detection

**EasyOLO** is a Python library designed to simplify the process of training and using YOLO (You Only Look Once) for object detection tasks. With a straightforward API, it allows users to easily load datasets, train YOLO models, and make predictions using images, directories, or even a webcam. This library is built on top of the **Ultralytics YOLO** implementation and streamlines the setup and usage of YOLO models.

## Benefits of EasyOLO

- **Ease of Use**: EasyOLO provides a clean and simple API for both training and inference.
- **Data Management**: Automatically splits data into training and validation sets, saving time and effort.
- **Versatile Predictions**: Perform predictions on single images, multiple images, or live webcam feeds with ease.
- **Customizable**: Fine-tune model training with customizable parameters like epochs, batch size, learning rate, and more.
- **Compatible**: Built on top of **Ultralytics YOLO**, one of the most popular YOLO implementations, ensuring high-quality results.

## Installation

To install EasyOLO, simply clone the repository or install it via pip:

```bash
pip install easyolo
```

## Quick Start Guide

### 1. Importing the Library

```python
from easyolo import EasyOLO
```

### 2. Initializing the `EasyOLO` Class

```python
yolo = EasyOLO()
```

### 3. Loading Data for Training

Use the `load_data` method to load your dataset for training. The method supports dataset splitting and validation set handling.

```python
yolo.load_data(
    image_dir='path/to/images',
    annotation_dir='path/to/annotations',
    validation=True,  # Set to True to use a validation set
    val_image_dir='path/to/validation/images',
    val_annotation_dir='path/to/validation/annotations',
    split=0.2,  # 20% of data for validation
    class_names=['class1', 'class2', 'class3']  # Optional, specify class names
)
```

### 4. Training the Model

After loading the data, you can train the YOLO model by calling the `train()` method. Specify the path to the data YAML file, as well as other hyperparameters such as epochs and learning rate.

```python
yolo.train(
    data_file='/content/data.yaml',
    epochs=100,
    batch=16,
    img_size=640,
    lr=0.01,
    save_dir='output/training',
    weights='/content/yolov5su.pt'  # Path to initial weights
)
```

### 5. Making Predictions

You can use the trained model to make predictions on single images, directories of images, or even from a webcam feed. 

- **Single Image Prediction**:

```python
yolo.predict(
    model_path='output/training/yolo_finetuned.pt',  # Path to trained model
    image_path='path/to/image.jpg'  # Path to an image
)
```

- **Multiple Images Prediction**:

```python
yolo.predict(
    model_path='output/training/yolo_finetuned.pt',
    image_dir='path/to/images'  # Directory containing images
)
```

- **Webcam Prediction**:

```python
yolo.predict(
    model_path='output/training/yolo_finetuned.pt',
    webcam_index=0  # Index of the webcam
)
```

### 6. Loading a Pre-trained Model

To use a pre-trained model, simply call the `load_model()` method:

```python
yolo.load_model('path/to/trained_model.pt')
```

## Key Methods

### `load_data()`

This method is used to load and prepare your dataset for training. You can specify directories for training and validation images, split data, and provide class names.

- **Arguments**:
  - `image_dir (str)`: Path to the images directory.
  - `annotation_dir (str)`: Path to the annotations directory.
  - `validation (bool)`: If `True`, use a separate validation set.
  - `val_image_dir (str)`: Path to the validation images directory (required if `validation=True`).
  - `val_annotation_dir (str)`: Path to the validation annotations directory (required if `validation=True`).
  - `split (float)`: Proportion of data to use for validation.
  - `class_names (list)`: List of class names for your dataset (optional).

### `train()`

This method is used to train the YOLO model. It requires a data YAML file and allows for customizing training parameters such as epochs, batch size, image size, learning rate, and more.

- **Arguments**:
  - `data_file (str)`: Path to the data YAML file.
  - `epochs (int)`: Number of epochs for training.
  - `batch (int)`: Batch size for training.
  - `img_size (int)`: Image size (default is 640).
  - `lr (float)`: Learning rate.
  - `save_dir (str)`: Directory to save the model and logs.
  - `weights (str)`: Path to the pre-trained weights file.

### `predict()`

This method performs predictions using a trained YOLO model. You can predict on a single image, a directory of images, or a live webcam feed.

- **Arguments**:
  - `model_path (str)`: Path to the trained YOLO model.
  - `image_path (str)`: Path to an image (optional if `image_dir` or `webcam_index` is provided).
  - `image_dir (str)`: Path to a directory of images (optional if `image_path` or `webcam_index` is provided).
  - `webcam_index (int)`: Index of the webcam for live prediction (optional).

### `load_model()`

This method is used to load a pre-trained YOLO model for prediction or further training.

- **Arguments**:
  - `model_path (str)`: Path to the pre-trained model.

## Example Workflow

Here is an example of the entire workflow:

1. **Prepare Data**:

```python
yolo.load_data(
    image_dir='images',
    annotation_dir='annotations',
    validation=True,
    val_image_dir='val_images',
    val_annotation_dir='val_annotations',
    split=0.2,
    class_names=['dog', 'cat', 'bird']
)
```

2. **Train the Model**:

```python
yolo.train(
    data_file='/content/data.yaml',
    epochs=50,
    batch=16,
    img_size=640,
    lr=0.01,
    save_dir='output/training',
    weights='/content/yolov5su.pt'
)
```

3. **Make Predictions**:

```python
yolo.predict(
    model_path='output/training/yolo_finetuned.pt',
    image_path='test.jpg'
)
```

4. **Webcam Predictions**:

```python
yolo.predict(
    model_path='output/training/yolo_finetuned.pt',
    webcam_index=0
)
```

## Conclusion

EasyOLO simplifies working with YOLO models for object detection tasks, making it easier for both beginners and experienced users to train and make predictions. Whether you're working with a single image or a live webcam feed, this library provides a simple and powerful API for your computer vision projects.
