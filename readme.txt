📁 AI Assist Vision

A modular AI-powered visual assistant that performs image captioning, object detection, and OCR (text recognition) with optional audio feedback. Built using powerful models like YOLOv8, BLIP-2, TrOCR, and EasyOCR.



📆 Features

- 🖼️ Scene description using BLIP-2
- 🎯 Object detection using YOLOv8
- 🔍 OCR using TrOCR and EasyOCR
- 🔊 Text-to-speech audio feedback (optional)
- 📸 Webcam support 



🗂️ Project Structure


ai_assist_vision/
│
├── main.py                  # Entry point for running the app
├── requirements.txt         # Python dependencies
│
├── modules/                 # Core functionalities
│   ├── image_capture.py     # Webcam and image input (Phase 2)
│   ├── object_detection.py  # YOLOv8 logic
│   ├── vlm_captioning.py    # Scene captioning with BLIP-2
│   ├── ocr_reader.py        # OCR with TrOCR + EasyOCR
│   └── audio_feedback.py    # Text-to-speech logic
│
└── utils/
    └── helpers.py           # Any helper functions



🚀 Getting Started

 1. Clone or Download

bash
git clone https://github.com/yourusername/ai_assist_vision.git
cd ai_assist_vision


Or simply unzip the folder if received as a `.zip`.


2. Create a Virtual Environment (Recommended)

bash
python -m venv venv
venv\Scripts\activate    # On Windows
# source venv/bin/activate  # On Linux/Mac




3. Install Dependencies

bash
pip install -r requirements.txt




4. Run the App

bash
python main.py



💠 Dependencies

Make sure you have the following models downloaded automatically on first use:

- `Salesforce/blip-image-captioning-base` (BLIP-2)
- `microsoft/trocr-base-printed` (TrOCR)
- YOLOv8 weights (`yolov8n.pt` by default from Ultralytics)



🖼️ Sample Input

Put test images inside the `sample_inputs/` folder (create it if not present). Modify `main.py` to point to your image.



🧑‍💻 Author
By Yash and Shruti

Feel free to use, improve, and share.

---

