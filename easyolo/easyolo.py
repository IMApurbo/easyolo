import os
import yaml
from yolov5 import train

class EasyOLO:
    def __init__(self):
        """
        Initialize the EasyOLO object
        """
        self.data_yaml = None
        self.model = None

    def load_data(self, image_dir: str, annotation_dir: str, validation=False, val_image_dir=None, val_annotation_dir=None, split=0.2):
        """
        Automatically loads and prepares image and annotation data for training and generates the data.yaml file.
        :param image_dir: Path to training images
        :param annotation_dir: Path to annotation files
        :param validation: If True, validation data is used from val_image_dir and val_annotation_dir
        :param val_image_dir: Path to validation images (if validation=True)
        :param val_annotation_dir: Path to validation annotations (if validation=True)
        :param split: Proportion of data used for validation (if validation=False)
        """
        # Prepare the data.yaml file
        data = {
            'train': image_dir,
            'val': val_image_dir if validation else image_dir,
            'nc': self._count_classes(annotation_dir),  # Number of classes based on annotations
            'names': self._get_class_names(annotation_dir)
        }

        # Save the data.yaml file to the working directory
        self.data_yaml = '/content/data.yaml'
        with open(self.data_yaml, 'w') as file:
            yaml.dump(data, file)

        print(f"Data loaded and data.yaml created at {self.data_yaml}")

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
        Train the YOLO model with custom data.
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

        # Begin training using the YOLOv5 train module
        train.run(
            data=self.data_yaml,             # Path to the data.yaml file
            img_size=img_size,               # Image size for training
            batch_size=batch_size,           # Training batch size
            epochs=epochs,                   # Number of epochs to train
            lr=lr,                           # Learning rate
            save_dir=save_dir,               # Directory to save the trained model and results
            weights=weights,                 # Path to pre-trained weights (default 'yolov5s.pt')
            device='0'                        # Use GPU if available
        )
        print(f"Training completed. The model has been saved to {save_dir}")
    
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

