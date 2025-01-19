# easyolo/train.py

from ultralytics import YOLO

def train(epochs: int=100, batch_size: int=16, img_size: int=640, lr: float=0.01, save_dir: str='output/training', weights: str='yolov5s.pt', **kwargs):
    """
    Train a YOLOv5 model using the given parameters.
    
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :param img_size: Image size.
    :param lr: Learning rate.
    :param save_dir: Directory to save the results.
    :param weights: Path to the pre-trained weights file.
    :param **kwargs: Additional parameters to pass to YOLOv5.
    """
    
    # Initialize the YOLO model with the pre-trained weights
    model = YOLO(weights)  # Load the pre-trained YOLOv5 model
    
    # Train the model
    model.train(
        data='data.yaml',  # Automatically generated data.yaml file
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size,
        project=save_dir,
        name='yolo_model',
        exist_ok=True,  # Overwrite existing folder
        hyp='data/hyp.scratch.yaml',  # Hyperparameter config
        lr0=lr,  # Learning rate
        **kwargs
    )
