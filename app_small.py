from flask import Flask, request, jsonify
from modules.object_detection import detect_objects
from PIL import Image
import io
import os

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    detections = detect_objects(temp_path)
    
    labels = [d['label'] for d in detections]

    return jsonify({
        "detected_objects": labels
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Important for Render
    app.run(host='0.0.0.0', port=port)
