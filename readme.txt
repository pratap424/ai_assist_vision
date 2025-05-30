This repository contains a modular, realtime assistive vision system that combines object detection, scene understanding, text reading (OCR), and audio feedback to help users — especially visually impaired individuals — understand their surroundings. It can also serve as a foundation for drone vision pipelines, smart surveillance systems, or any visual intelligence interface.



├── modules/
│   ├── object\_detection.py      YOLOv8 object detection
│   ├── vlm\_captioning.py        Scene captioning using BLIP
│   ├── ocr\_reader.py            OCR using TrOCR + EasyOCR
│   ├── audio\_feedback.py        Texttospeech system
│   └── camera\_capture.py        Webcam image capture





 🚀 Key Features

 🧭 Scene Captioning: Understand and describe what is happening in an image using BLIP.
 🧱 Object Detection: Identify multiple objects with YOLOv8.
 📝 Text Extraction: Extract readable text using a hybrid of TrOCR (transformerbased OCR) and EasyOCR.
 🔊 Audio Feedback: Speak out the combined information using offline TTS (pyttsx3).
 🎥 Live Camera Capture: Take pictures from your webcam in real time.



 🧠 Module Descriptions

1️⃣ object_detection.py – Object Detector

 Uses YOLOv8 (nano variant) to detect objects from the input image.
 Outputs a unique list of class names (e.g., [person, car, bicycle]).

python
from modules.object_detection import detect_objects
objects = detect_objects("image.jpg")




 2️⃣ vlm_captioning.py – Scene Caption Generator

Uses BLIP (Salesforce) for generating natural language descriptions of the full scene.

python
from modules.vlm_captioning import describe_scene
scene = describe_scene("image.jpg")


> Model: Salesforce/blipimagecaptioningbase



 3️⃣ ocr_reader.py – Text Reader (OCR)

Uses TrOCR from HuggingFace and EasyOCR together for better printedtext recognition.
TrOCR handles structured text; EasyOCR complements with flexible scene text detection.

python
from modules.ocr_reader import read_text_combined
text = read_text_combined("image.jpg")


> Models:
> microsoft/trocrbaseprinted + EasyOCR (English)



 4️⃣ audio_feedback.py – Speech Output

Uses pyttsx3, an offline TTS engine, to vocalize any given string.
Helps visually impaired users hear the output.

python
from modules.audio_feedback import speak
speak("Hello world")




 5️⃣ camera_capture.py – Webcam Image Capture

Opens a webcam feed using OpenCV.
Press c to capture an image, or q to quit.

python
from modules.camera_capture import capture_image
image_path = capture_image()




 🎯 main.py – Integrated Inference Pipeline

This is the master script that connects everything:

 ✅ What It Does

1. Loads an image (default: sample_inputs/test_image.jpg)
2. Describes the scene using BLIP
3. Detects objects using YOLOv8
4. Extracts text using OCR
5. Speaks all results aloud using TTS

bash
python main.py


 🖼️ Sample Output:


🔍 Analyzing: sample_inputs/test_image.jpg

🖼️ Scene: A man riding a bicycle on a street.
📦 Objects: [person, bicycle, car]
📝 Text: Caution. Construction Zone Ahead.

🔊 Speaking...
I see a man riding a bicycle on a street. I detected the following objects: person, bicycle, car. The text says: Caution. Construction Zone Ahead.


You can change the image being analyzed by modifying the path in main.py:

python
run_pipeline("your_image.jpg")




 📦 Installation

 🔧 Dependencies

Install all required packages using pip:

bash
pip install torch torchvision torchaudio
pip install transformers
pip install easyocr
pip install pyttsx3
pip install opencvpython
pip install ultralytics




 📸 Sample Image Setup

Place your input image inside a sample_inputs/ folder, or capture it live using camera_capture.py.



 🎯 Use Cases

* 👁️‍🗨️ Assisting visually impaired users
* 🚁 Visionlanguage systems for drones
* 🧠 Multimodal research in VLMs
* 🛑 Road sign/text extraction
* 📷 Scene understanding for surveillance



 📈 Future Improvements

* Offline deployment via ONNX or TFLite
* Android/iOS app integration
* Multilingual speech output
* Navigationaware assistance using depth and segmentation




