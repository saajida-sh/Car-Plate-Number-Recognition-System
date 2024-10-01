# Car-Plate-Number-Recognition-System

This project aims to build an AI-powered system to automatically detect and recognize car number plates from images. The system is designed to first identify the license plate's location within the image, and then extract the number on the plate.

Key Features:

- Plate Detection: Using bounding boxes to identify the car plate region from images.
- Number Recognition: Leveraging machine learning models to recognize and decode the license plate number.
- Preprocessing: The images undergo resizing and normalization, and annotations are parsed from XML files in PASCAL VOC format.
- Modeling Approach: The project uses both neural network-based models and API fine-tuning for plate detection. Initial attempts involve training on the InceptionResNetV2 architecture.
- Dataset: The dataset includes 453 labeled images with corresponding bounding box annotations for car plates, stored in the PASCAL VOC format.
