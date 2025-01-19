import os
import cv2
from ultralytics import YOLO

class EasyOLO:
    def __init__(self):
        """Initialize the EasyOLO object."""
        self.model = None
    
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

