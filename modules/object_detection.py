from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

def detect_objects(image_path, conf_threshold=0.4):
    results = yolo_model(image_path)[0]
    names = results.names

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls[0].item())
        label = names[cls_id]
        bbox = [int(coord) for coord in box.xyxy[0].tolist()]

        # Get confidence safely
        try:
            confidence = float(box.conf[0])
        except:
            confidence = 1.0  # fallback if confidence isn't provided

        if confidence >= conf_threshold:
            detections.append({
                "label": label,
                "bbox": bbox,
                "confidence": round(confidence, 2)
            })

    return detections

