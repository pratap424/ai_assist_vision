from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = yolo_model(image_path)[0]
    names = results.names
    classes = results.boxes.cls.tolist()
    return list(set([names[int(cls)] for cls in classes]))
