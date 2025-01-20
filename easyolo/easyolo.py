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
        self.data_yaml = None
        self.model = None

    def load_data(self, image_dir, annotation_dir, validation=False, val_image_dir=None, val_annotation_dir=None, split=0.2):
        train_image_dir = Path(image_dir) / 'train'
        val_image_dir = Path(image_dir) / 'val'
        train_annotation_dir = Path(annotation_dir) / 'train'
        val_annotation_dir = Path(annotation_dir) / 'val'

        if validation:
            if not (val_image_dir and val_annotation_dir):
                raise ValueError("Validation directories must be provided when validation=True.")
        else:
            self._split_data(image_dir, annotation_dir, split, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir)

        data = {
            'train': str(train_image_dir),
            'val': str(val_image_dir),
            'nc': self._count_classes(annotation_dir),
            'names': self._get_class_names(annotation_dir),
        }

        self.data_yaml = '/content/data.yaml'
        with open(self.data_yaml, 'w') as file:
            yaml.dump(data, file)
        print(f"Data loaded and data.yaml created at {self.data_yaml}")

    def _split_data(self, image_dir, annotation_dir, split, train_image_dir, val_image_dir, train_annotation_dir, val_annotation_dir):
        train_image_dir.mkdir(parents=True, exist_ok=True)
        val_image_dir.mkdir(parents=True, exist_ok=True)
        train_annotation_dir.mkdir(parents=True, exist_ok=True)
        val_annotation_dir.mkdir(parents=True, exist_ok=True)

        images = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        random.shuffle(images)
        split_idx = int(len(images) * (1 - split))

        for img in images[:split_idx]:
            shutil.copy(img, train_image_dir / img.name)
            annotation = annotation_dir / img.with_suffix('.txt').name
            shutil.copy(annotation, train_annotation_dir / annotation.name)

        for img in images[split_idx:]:
            shutil.copy(img, val_image_dir / img.name)
            annotation = annotation_dir / img.with_suffix('.txt').name
            shutil.copy(annotation, val_annotation_dir / annotation.name)

    def _count_classes(self, annotation_dir):
        class_ids = set()
        for file in Path(annotation_dir).glob('*.txt'):
            with file.open() as f:
                for line in f:
                    class_ids.add(line.split()[0])
        return len(class_ids)

    def _get_class_names(self, annotation_dir):
        class_ids = set()
        for file in Path(annotation_dir).glob('*.txt'):
            with file.open() as f:
                for line in f:
                    class_ids.add(line.split()[0])
        return sorted(class_ids)

    def train(self, epochs=100, batch_size=16, img_size=640, lr=0.01, save_dir='output/training', weights='yolov5s.pt', custom_data_file=None):
        if custom_data_file:
            if not Path(custom_data_file).exists():
                raise FileNotFoundError(f"Custom data file {custom_data_file} not found.")
            self.data_yaml = custom_data_file
        elif not self.data_yaml:
            raise ValueError("Data.yaml file not found. Please load data first or provide a custom data file.")

        self.model = YOLO(weights)
        self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,
            lr0=lr,
            project=save_dir,
            name='yolo_finetuned',
            exist_ok=True,
        )
        print(f"Training completed. Model saved at {save_dir}/yolo_finetuned")

    def predict(self, model_path, image_path=None, image_dir=None, webcam_index=None):
        if not self.model:
            self.load_model(model_path)

        if image_path:
            self._predict_single_image(image_path)
        elif image_dir:
            self._predict_multiple_images(image_dir)
        elif webcam_index is not None:
            self._predict_webcam(webcam_index)
        else:
            print("Provide an image path, directory, or webcam index for prediction.")

    def load_model(self, model_path):
        if Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

    def _predict_single_image(self, image_path):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image {image_path} does not exist.")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image {image_path}.")
        results = self.model(image)
        results[0].plot()

    def _predict_multiple_images(self, image_dir):
        for image_path in Path(image_dir).glob('*.[jp][pn]g'):
            self._predict_single_image(str(image_path))

    def _predict_webcam(self, webcam_index):
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {webcam_index}.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            results[0].plot()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
