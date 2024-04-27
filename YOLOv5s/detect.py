import numpy as np
from PIL import Image
from yolov5 import YOLOv5

model = YOLOv5('yolov5s.pt')

def detect_and_crop(image_path):
    image = Image.open(image_path).convert("RGB")
    results = model.predict(np.array(image))
    detections = results.pandas().xyxy[0]  # Extract bounding boxes as DataFrame

    crops = []
    details = []
    image_np = np.array(image)  # Convert PIL image to numpy array

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']  # Extract confidence score
        class_id = row['class']  # Extract class ID
        class_name = row['name']  # Extract class name if available, or use model.names[class_id] if not

        crop = image_np[y1:y2, x1:x2]
        crops.append(crop)
        details.append((class_id, class_name, conf, (x1, y1, x2, y2)))  # Append all details as a tuple

    return crops, details, image_np