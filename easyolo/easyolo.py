import os
import shutil
import yaml
import cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

class EasyOLO:
    def __init__(self, output_dir='output/training'):
        """Initialize the EasyOLO object."""
        self.output_dir = output_dir
        self.model = None
        self.data_file = None

    def load_data(self, image_dir, annotation_dir, validation=False, split=0.2, val_image_dir=None, val_annotation_dir=None):
        """
        Load the data and split it if validation is false, otherwise, use the provided validation directories.
        Automatically creates a data.yaml file needed for training.
        """
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations', 'val'), exist_ok=True)

        if validation:
            # Use the provided validation image and annotation directories
            self.val_image_dir = val_image_dir
            self.val_annotation_dir = val_annotation_dir
        else:
            # Split the data into train and validation sets if no validation directories are given
            train_images, val_images = train_test_split(os.listdir(image_dir), test_size=split)
            # Move validation images to output folder
            for val_image in val_images:
                shutil.move(os.path.join(image_dir, val_image), os.path.join(self.output_dir, 'images', 'val', val_image))
                shutil.move(os.path.join(annotation_dir, val_image.replace('.jpg', '.xml')), os.path.join(self.output_dir, 'annotations', 'val', val_image.replace('.jpg', '.xml')))
            # Set up paths for training images and annotations
            self.train_image_dir = os.path.join(image_dir, 'train')
            self.train_annotation_dir = os.path.join(annotation_dir, 'train')

        # Create data.yaml
        self._create_data_yaml(image_dir, annotation_dir)

    def _create_data_yaml(self, image_dir, annotation_dir):
        """
        Create data.yaml file for YOLO training, specifying paths and labels.
        """
        class_names = self._get_class_names(annotation_dir)
        data = {
            'train': os.path.join(self.output_dir, 'images', 'train'),
            'val': os.path.join(self.output_dir, 'images', 'val'),
            'nc': len(class_names),
            'names': class_names
        }
        
        self.data_file = os.path.join(self.output_dir, 'data.yaml')
        with open(self.data_file, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
    
    def _get_class_names(self, annotation_dir):
        """
        Extract class names from annotation files (assuming XML format for simplicity).
        """
        # Assuming each annotation file contains a single object type
        class_names = set()
        for annotation in os.listdir(annotation_dir):
            if annotation.endswith('.xml'):
                # Read the XML file and extract the class name
                # For simplicity, we're just extracting the filename
                class_names.add(annotation.split('_')[0])  # This needs modification based on your XML structure
        return list(class_names)
    
    def train(self, epochs=100, batch_size=16, img_size=640, pretrained=False):
        """
        Train the YOLO model with the loaded dataset and specified hyperparameters.
        """
        # Initialize model and start training
        if not self.model:
            self.model = YOLO()  # Initialize YOLO model
        
        self.model.train(data=self.data_file, epochs=epochs, batch_size=batch_size, imgsz=img_size, pretrained=pretrained)
    
    def load_model(self, model_path):
        """Load the YOLO model from the specified path."""
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model path {model_path} does not exist.")

    def predict(self, model_path, image_path=None, image_dir=None, webcam_index=None):
        """Make predictions using the specified YOLO model."""
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

    def _predict_single_image(self, image_path):
        """Process a single image and display predictions."""
        image = cv2.imread(image_path)
        results = self.model(image)
        results.show()

    def _predict_multiple_images(self, image_dir):
        """Process all images in a directory and display predictions."""
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                self._predict_single_image(image_path)

    def _predict_webcam(self, webcam_index):
        """Stream video from the webcam and show predictions."""
        cap = cv2.VideoCapture(webcam_index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            results.show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
