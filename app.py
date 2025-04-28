from flask import Flask, request, jsonify
from main import run_pipeline
from PIL import Image
import io

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    # ✅ Call your pipeline and capture result
    output = run_pipeline(image_path=temp_path, save_output=False, speak_enabled=False)

    # ✅ Return JSON
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
