import os
import torch
import cv2
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

class EasyOLO:
    def __init__(self):
        """
        Initialize the EasyOLO object
        """
        self.data_yaml = None
        self.model = None

    def load_data(self, image_dir: str, annotation_dir: str, validation=False, val_image_dir=None, val_annotation_dir=None, split=0.2):
        """
        Automatically load and prepare image and annotation data for training and generates the data.yaml file.
        :param image_dir: Path to training images
        :param annotation_dir: Path to annotation files
        :param validation: If True, validation data is used from val_image_dir and val_annotation_dir
        :param val_image_dir: Path to validation images (if validation=True)
        :param val_annotation_dir: Path to validation annotations (if validation=True)
        :param split: Proportion of data used for validation (if validation=False)
        """
        # Create the train/val directories if they don't exist
        train_image_dir = Path(image_dir) / 'train'
        val_image_dir = Path(image_dir) / 'val'
        train_annotation_dir = Path(annotation_dir) / 'train'
        val_annotation_dir = Path(annotation_dir) / 'val'
        
        if not validation:
            # Split dataset into train/val based on the split ratio
            self._split_data(image_dir, annotation_dir, split, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir)
        
        # Prepare the data.yaml file
        data = {
            'train': str(train_image_dir),
            'val': str(val_image_dir),
            'nc': self._count_classes(annotation_dir),  # Number of classes based on annotations
            'names': self._get_class_names(annotation_dir)
        }

        # Save the data.yaml file to the working directory
        self.data_yaml = '/content/data.yaml'
        with open(self.data_yaml, 'w') as file:
            yaml.dump(data, file)

        print(f"Data loaded and data.yaml created at {self.data_yaml}")

    def _split_data(self, image_dir, annotation_dir, split, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir):
        """
        Split data into training and validation sets.
        :param image_dir: Path to training images
        :param annotation_dir: Path to annotation files
        :param split: Proportion of data for validation
        :param train_image_dir: Directory for training images
        :param val_image_dir: Directory for validation images
        :param train_annotation_dir: Directory for training annotations
        :param val_annotation_dir: Directory for validation annotations
        """
        # Create directories for train and validation if they don't exist
        train_image_dir.mkdir(parents=True, exist_ok=True)
        val_image_dir.mkdir(parents=True, exist_ok=True)
        train_annotation_dir.mkdir(parents=True, exist_ok=True)
        val_annotation_dir.mkdir(parents=True, exist_ok=True)

        # Get a list of all image files and shuffle them
        images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)

        # Split based on the split ratio
        split_idx = int(len(images) * (1 - split))

        # Move files into respective directories
        for img in images[:split_idx]:
            shutil.copy(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
            anno_file = img.replace(img.split('.')[-1], 'txt')
            shutil.copy(os.path.join(annotation_dir, anno_file), os.path.join(train_annotation_dir, anno_file))

        for img in images[split_idx:]:
            shutil.copy(os.path.join(image_dir, img), os.path.join(val_image_dir, img))
            anno_file = img.replace(img.split('.')[-1], 'txt')
            shutil.copy(os.path.join(annotation_dir, anno_file), os.path.join(val_annotation_dir, anno_file))

    def _count_classes(self, annotation_dir: str):
        """
        Count the number of classes in the annotation directory.
        :param annotation_dir: Directory containing annotation files
        :return: Number of unique classes
        """
        class_names = set()
        for file in os.listdir(annotation_dir):
            if file.endswith('.txt'):
                with open(os.path.join(annotation_dir, file)) as f:
                    labels = [line.split()[0] for line in f.readlines()]
                    class_names.update(labels)
        return len(class_names)

    def _get_class_names(self, annotation_dir: str):
        """
        Get class names from the annotation directory.
        :param annotation_dir: Directory containing annotation files
        :return: List of class names
        """
        class_names = set()
        for file in os.listdir(annotation_dir):
            if file.endswith('.txt'):
                with open(os.path.join(annotation_dir, file)) as f:
                    labels = [line.split()[0] for line in f.readlines()]
                    class_names.update(labels)
        return sorted(class_names)

    def train(self, epochs=100, batch_size=16, img_size=640, lr=0.01, save_dir='output/training', weights='yolov5s.pt'):
        """
        Train the YOLO model with custom data using the ultralytics YOLO module.
        :param epochs: Number of epochs for training
        :param batch_size: Batch size for training
        :param img_size: Image size for training
        :param lr: Learning rate
        :param save_dir: Directory to save results
        :param weights: Pre-trained weights file path
        """
        # Ensure the data.yaml file is present
        if not self.data_yaml:
            raise ValueError("Data.yaml file not found. Please load data first using the load_data() method.")

        # Load the YOLO model using pre-trained weights
        self.model = YOLO(weights)  # 'yolov5s.pt' by default, or other pretrained weights

        # Start the training process using ultralytics YOLO module
        self.model.train(
            data=self.data_yaml,          # Path to the data.yaml file
            epochs=epochs,                # Number of epochs
            batch_size=batch_size,        # Batch size
            imgsz=img_size,               # Image size for training
            lr0=lr,                       # Learning rate
            project=save_dir,             # Directory to save the training output
            name='yolo_finetuned',        # Name of the experiment
            exist_ok=True                 # Allow overwriting of results
        )
        print(f"Training completed. The model has been saved to {save_dir}/yolo_finetuned")
    
    def predict(self, model_path, image_path=None, image_dir=None, webcam_index=None):
        """
        Make predictions using the specified YOLO model.
        - model_path: Path to the trained YOLO model.
        - image_path: Path to a single image for prediction.
        - image_dir: Directory containing images for prediction.
        - webcam_index: Index of the webcam for live predictions.
        """
        # Load the model if it's not loaded
        if not self.model:
            self.load_model(model_path)

        if image_path:
            self._predict_single_image(image_path)
        
        elif image_dir:
            self._predict_multiple_images(image_dir)
        
        elif webcam_index is not None:
            self._predict_webcam(webcam_index)
        else:
            print("Please provide either an image path, image directory, or webcam index for prediction.")

    def load_model(self, model_path):
        """Load the YOLO model from the specified path."""
        if os.path.exists(model_path):
            self.model = YOLO(model_path)  # Load model from local path
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model path {model_path} does not exist.")
    
    def _predict_single_image(self, image_path):
        """Process a single image and display predictions."""
        print(f"Processing image: {image_path}")
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} does not exist.")
            return

        # Read and process the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}.")
            return

        results = self.model(image)

        # Handle results, ensuring it's a single result
        if isinstance(results, list):
            results = results[0]  # Extract the first result if it's a list

        results.show()  # Show results on the image

    def _predict_multiple_images(self, image_dir):
        """Process all images in a directory and display predictions."""
        print(f"Processing images in directory: {image_dir}")
        
        # Check if the directory exists
        if not os.path.isdir(image_dir):
            print(f"Error: Directory {image_dir} does not exist.")
            return

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                self._predict_single_image(image_path)

    def _predict_webcam(self, webcam_index):
        """Stream video from the webcam and show predictions."""
        print(f"Streaming webcam video on index {webcam_index}")
        
        # Open webcam
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {webcam_index}.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make predictions on the frame
            results = self.model(frame)

            # Handle results, ensuring it's a single result
            if isinstance(results, list):
                results = results[0]  # Extract the first result if it's a list

            results.show()  # Show the results on the frame
            
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows

