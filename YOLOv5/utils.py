from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

model = YOLOv5('yolov5x.pt')

def load_image(image_path):
    # Load an image and convert it to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Resize image maintaining aspect ratio
    original_width, original_height = image.size
    max_size = 640
    scale = min(max_size / original_width, max_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Pad the resized image to be 640x640
    new_image = Image.new('RGB', (max_size, max_size))
    new_image.paste(image, ((max_size - new_width) // 2, (max_size - new_height) // 2))
    
    return np.array(new_image)  # Convert PIL image to numpy array for processing

def detect_objects(image_path):
    image = load_image(image_path)
    results = model.predict(image)
    return results

def draw_boxes(image_np, detections):
    """
    Draw bounding boxes with labels on the image.
    """
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index, row in detections.iterrows():
        # Draw rectangle to highlight the object
        draw.rectangle([(row['xmin'], row['ymin']), (row['xmax'], row['ymax'])], outline="red", width=2)
        # Prepare label text with class name and confidence score
        label = f"{row['name']} {row['confidence']:.2f}"
        # Draw text on top of the rectangle
        draw.text((row['xmin'], row['ymin']), label, fill="red", font=font)

    return np.array(image)

def save_detections(image_path, results, output_folder):
    """
    Save the detection results, including images with drawn bounding boxes and details in a text file.
    """
    # Make sure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load and prepare image
    image_np = load_image(image_path)  # Load and resize the image properly
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image_np, detections)

    # Save the annotated image
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    Image.fromarray(image_with_boxes).save(output_image_path)

    # Save detection details to a text file
    output_text_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_detections.txt')
    with open(output_text_path, 'w') as f:
        for index, row in detections.iterrows():
            f.write(f"Class ID: {row['class']}, Class Name: {row['name']}, Confidence: {row['confidence']:.2f}, BBox: [{row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}]\n")

    print(f"Results saved: {output_image_path} and {output_text_path}")

