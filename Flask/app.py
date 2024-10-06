from flask import Flask, request, render_template
import os
import cv2
import easyocr
import torch
import string
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_trained_model/weights/best.pt', force_reload=True)  # Adjust the path to your model
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the image
        original_image, result_image, ocr_texts = process_image(file_path)
        
        # Save the result images
        original_image_path = os.path.join(UPLOAD_FOLDER, 'original_' + file.filename)
        result_image_path = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
        
        original_image.save(original_image_path)
        result_image.save(result_image_path)

        return render_template('index.html', original_image='original_' + file.filename, result_image='result_' + file.filename, ocr_texts=ocr_texts)

    return render_template('index.html')

def clean_text(text):
    """Clean the detected text by removing punctuation and converting to uppercase."""
    # Remove punctuation and convert to uppercase
    cleaned_text = text.translate(str.maketrans("", "", string.punctuation)).strip().upper()
    return cleaned_text

def resize_image(image, max_width=600, max_height=450):
    """Resize the image to fit within the specified max width and height."""
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def process_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform inference with YOLO
    results = model(frame)
    predictions = results.xyxy[0]  # Get predictions

    ocr_texts = []  # Store OCR results

    # Loop through each detected bounding box
    for *box, conf, cls in predictions:
        x1, y1, x2, y2 = map(int, box)  # Convert to integers

        # Crop the region of interest for OCR
        roi = frame[y1:y2, x1:x2]
        ocr_results = reader.readtext(roi)

        # Draw a single bounding box around the license plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

        # Draw bounding box and detected text
        for (bbox, text, prob) in ocr_results:
            
            ocr_texts.append(clean_text(text))
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Adjust the coordinates of the detected text back to the original frame
            adjusted_top_left = (top_left[0] + x1, top_left[1] + y1)
            adjusted_bottom_right = (bottom_right[0] + x1, bottom_right[1] + y1)
            
            # Put the detected text on the frame
            cv2.putText(frame, clean_text(text), (adjusted_top_left[0], adjusted_top_left[1] - 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

    # Convert the frame to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert back to PIL image for saving
    result_image = Image.fromarray(frame_rgb)
    
     # Resize images to fit within specified dimensions
    original_image_resized = resize_image(image)
    result_image_resized = resize_image(result_image)

    return original_image_resized, result_image_resized, ocr_texts

if __name__ == '__main__':
    app.run(debug=True)
