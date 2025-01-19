from ultralytics import YOLO
import os
import yaml
import shutil
from sklearn.model_selection import train_test_split

class EasyOLO:
    def __init__(self, output_dir='output/training'):
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
    
    def predict(self, source, show=True, save=False):
        """
        Perform inference on an image, a directory of images, or a webcam.
        """
        if not self.model:
            raise ValueError("Model is not trained yet.")
        
        # Perform prediction on the source (image path, directory, or webcam)
        results = self.model(source)
        
        if show:
            results.show()  # Display the image with bounding boxes
        
        if save:
            results.save()  # Save the results to output
        
        return results
