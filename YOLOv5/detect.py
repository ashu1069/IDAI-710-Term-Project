from utils import load_image
from yolov5 import YOLOv5
from utils import save_detections, detect_objects

model = YOLOv5('yolov5x.pt')


image_path = 'path_to_your_image.jpg'
output_folder = 'detected_results'
results = detect_objects(image_path)
save_detections(image_path, results, output_folder)