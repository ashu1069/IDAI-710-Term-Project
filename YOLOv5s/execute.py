from detect import detect_and_crop
import cv2

image_path = 'path to images'
crops, details, preprocessed_image = detect_and_crop(image_path)

for idx, (crop, detail) in enumerate(zip(crops, details)):
    class_id, class_name, conf, bbox = detail
    print(f"Crop {idx+1}: Class ID={class_id}, Class Name={class_name}, Confidence={conf:.2f}, BBox={bbox}")
    save_path = f'.../crop_{idx + 1}_{class_name}_{int(conf * 100)}.png'
    cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
