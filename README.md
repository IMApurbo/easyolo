# EasyOLO: Simplified YOLO Fine-Tuning & Uses

EasyOLO is a Python package designed to make the process of fine-tuning YOLO (You Only Look Once) models easier for object detection tasks. It simplifies the loading of data, automatic generation of the `data.yaml` file, and provides methods for training and inference. This is ideal for users who want a straightforward way to fine-tune YOLO models without worrying about complex setup and configuration.

## Installation

To install `easyolo`, you can use the following command:

```bash
pip install easyolo
```

Or, if you prefer to install directly from the GitHub repository:

```bash
git clone https://github.com/your-username/easyolo.git
cd easyolo
pip install .
```

## Features

- **Data Loading**: Automatically loads and prepares image and annotation data for training, with support for train-validation split.
- **`data.yaml` Creation**: Automatically generates the required `data.yaml` file for YOLOv5, including class labels and the number of classes.
- **Training**: Simplifies the process of training YOLOv5 by passing configuration options directly into the function.
- **Inference**: Perform object detection using a pre-trained model with easy image or webcam input.

## Usage

### 1. **Load Data**

The first step is to load the image and annotation data using the `easyolo.load_data()` method. This will automatically generate the `data.yaml` file used for training.

#### Syntax:

```python
easyolo.load_data(image_dir: str, annotation_dir: str, validation: bool=False, val_image_dir: str=None, val_annotation_dir: str=None, split: float=0.2)
```

#### Parameters:
- `image_dir`: Path to the directory containing training images.
- `annotation_dir`: Path to the directory containing annotation files (YOLO `.txt` format).
- `validation`: Set to `True` if you have separate validation data. Default is `False`.
- `val_image_dir`: Path to the validation image directory (required if `validation=True`).
- `val_annotation_dir`: Path to the validation annotation directory (required if `validation=True`).
- `split`: The proportion of data to use for validation if `validation=False`. Default is `0.2`.

#### Example 1: **Loading Data with Train-Validation Split**

```python
import easyolo

# Load the training and validation data
easyolo.load_data(
    image_dir='/images',             # Path to training images
    annotation_dir='/annotations',   # Path to annotations (YOLO format)
    validation=True,                 # Enable validation set
    val_image_dir='/val_images',     # Path to validation images
    val_annotation_dir='/val_annotations'  # Path to validation annotations
)
```

#### Example 2: **Loading Data with a Train Split (Default Split = 0.2)**

```python
import easyolo

# Load the training data with a 20% validation split
easyolo.load_data(
    image_dir='/images',             # Path to training images
    annotation_dir='/annotations',   # Path to annotations (YOLO format)
    validation=False,                # No separate validation data
    split=0.2                        # Use 20% for validation
)
```

### 2. **Train the Model**

Once the data is loaded, you can easily train the model using `easyolo.train()`.

#### Syntax:

```python
easyolo.train(epochs: int=100, batch_size: int=16, img_size: int=640, lr: float=0.01, save_dir: str='output/training', weights: str='yolov5s.pt', **kwargs)
```

#### Parameters:
- `epochs`: Number of epochs for training. Default is `100`.
- `batch_size`: Batch size for training. Default is `16`.
- `img_size`: Size of the images used for training. Default is `640`.
- `lr`: Learning rate. Default is `0.01`.
- `save_dir`: Directory to save the trained model and results. Default is `'output/training'`.
- `weights`: Path to the pre-trained weights file. Default is `'yolov5s.pt'`.
- `**kwargs`: Any other YOLOv5-specific training parameters.

#### Example 1: **Training the Model**

```python
import easyolo

# Train the model using the automatically generated data.yaml
easyolo.train(
    epochs=100,                             # Number of epochs
    batch_size=16,                          # Batch size
    img_size=640,                           # Image size
    lr=0.01,                                # Learning rate
    save_dir='output/training'              # Save results to this directory
)
```

#### Example 2: **Training the Model with Custom Hyperparameters**

```python
import easyolo

# Train the model using custom hyperparameters
easyolo.train(
    epochs=200,                             # Number of epochs
    batch_size=32,                          # Batch size
    img_size=416,                           # Image size
    lr=0.005,                               # Learning rate
    save_dir='output/training/custom_model' # Save results to this directory
)
```

### 3. **Inference (Object Detection)**

Once the model is trained, you can perform inference (object detection) on an image, directory of images, or even using a webcam.

#### Syntax:

```python
easyolo.detect(input_type: str, input_path: str, model_path: str='output/training/weights/best.pt', **kwargs)
```

#### Parameters:
- `input_type`: Type of input for inference. Can be `'image'`, `'images'`, or `'webcam'`.
- `input_path`: Path to the image or directory of images, or camera number for webcam.
- `model_path`: Path to the trained model weights. Default is `'output/training/weights/best.pt'`.
- `**kwargs`: Additional parameters for inference (e.g., confidence threshold, image size).

#### Example 1: **Image Inference**

```python
import easyolo

# Perform object detection on a single image
easyolo.detect(
    input_type='image',          # Single image
    input_path='/path/to/image.jpg',  # Path to the image
    model_path='output/training/weights/best.pt'  # Path to the trained model
)
```

#### Example 2: **Directory Inference**

```python
import easyolo

# Perform object detection on a directory of images
easyolo.detect(
    input_type='images',              # Multiple images
    input_path='/path/to/images/',    # Directory containing images
    model_path='output/training/weights/best.pt'  # Path to the trained model
)
```

#### Example 3: **Webcam Inference**

```python
import easyolo

# Perform object detection using the webcam (e.g., camera 0)
easyolo.detect(
    input_type='webcam',             # Webcam input
    input_path='0',                  # Camera number (e.g., 0)
    model_path='output/training/weights/best.pt'  # Path to the trained model
)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the original YOLOv5 implementation.
- [PyTorch](https://pytorch.org/) for deep learning framework.

