# Car License Plate Recognition System

## Project Overview

This project implements a car license plate recognition system using YOLO (You Only Look Once) for object detection and EasyOCR for optical character recognition (OCR). 

Key Features:

- Dataset: The dataset consists of 658 labeled images with corresponding bounding box annotations for car license plates, sourced from two publicly available datasets on Kaggle and stored in PASCAL VOC format.
- Preprocessing: The images are resized and normalized, while the annotations are parsed from XML files in PASCAL VOC format.
- Plate Detection: Utilizes a trained YOLO model, which is an efficient object detection model for real-time applications.
- Number Recognition: Implements EasyOCR, a neural network-based model for accurate text recognition.

## Contents
- **dataset_yolo_formatting.ipynb**: A notebook for formatting the dataset for YOLO training.
- **yolo_training.ipynb**: A notebook that contains the implementation and training of the YOLO model.
- **yolo_recognition_model.ipynb**: A notebook for making predictions on both images and videos.

## Technologies Used
- **YOLOv5**: For object detection of car license plates.
- **EasyOCR**: For recognizing characters from detected license plates.
- **OpenCV**: For image processing and handling.
- **Python**: Programming language used for implementation.

## Installation
To run this project, you need to have the following packages installed:
```bash
pip install torch torchvision torchaudio  # For YOLOv5
pip install easyocr opencv-python ipywidgets  # For OCR and image processing

