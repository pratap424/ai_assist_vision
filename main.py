import os

# IMPORTANT: Force matplotlib to non-GUI backend if running on Render
if os.environ.get("RENDER", "0") == "1":
    import matplotlib
    matplotlib.use('Agg')

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt


import cv2


import sys


# Add the main project directory to the Python path
sys.path.append(os.path.abspath("."))  # You can replace "." with your project path

from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak
from modules.camera_capture import capture_image
#from modules.qna_module import start_qna

# ... after you have an image_path ...
#start_qna(image_path)


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
from collections import Counter

from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak
from modules.camera_capture import capture_image


def run_pipeline(image_path=None, save_output=False, output_path="output.txt",speak_enabled=True):
    # Capture from webcam if no image path is provided
    if image_path is None:
        image_path = capture_image(use_gui=True)
    print("🔍 Analyzing:", image_path)

    # Run modules
    scene = describe_scene(image_path)
    detections = detect_objects(image_path)
    text = read_text_combined(image_path)

    # Count objects
    labels = [d["label"] for d in detections]
    counts = Counter(labels)

    # Determine spatial regions and collect detections
    img = Image.open(image_path)
    W, H = img.size
    region_dets = {"left": [], "center": [], "right": []}
    for d in detections:
        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
        cx = (x1 + x2) / 2
        if cx < W/3:
            region = "left"
        elif cx > 2*W/3:
            region = "right"
        else:
            region = "center"
        region_dets[region].append(d)

    # Build basic count phrases
    obj_phrases = [f"{cnt} {lbl}{'s' if cnt>1 else ''}" for lbl, cnt in counts.items()]
    region_phrases = []
    for reg, dets in region_dets.items():
        ctr = Counter([d["label"] for d in dets])
        for lbl, cnt in ctr.items():
            region_phrases.append(f"{cnt} {lbl}{'s' if cnt>1 else ''} on the {reg}")

    # Detect person-on-vehicle relationships (e.g., person on motorcycle)
    rel_phrases = []
    for reg, dets in region_dets.items():
        persons = [d for d in dets if d["label"] == "person"]
        vehicles = [d for d in dets if d["label"] in ("motorcycle", "bicycle", "motorbike", "bike")]
        for p in persons:
            px1, py1, px2, py2 = p.get("bbox", [0,0,0,0])
            pcx, pcy = (px1+px2)/2, (py1+py2)/2
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v.get("bbox", [0,0,0,0])
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    rel_phrases.append(f"a person on a {v['label']} on the {reg}")
                    break

    # Construct full descriptive output
    rel_part = ''
    if rel_phrases:
        rel_part = ' '.join([f"There is {rp}." for rp in rel_phrases]) + ' '
    final_output = (
        f"{rel_part}The image shows {scene}. "
        f"I see {', '.join(obj_phrases)}. "
        f"Specifically, {', '.join(region_phrases)}. "
        #f"The visible text reads: '{text}'."
    )

    # Print & speak
    print("\n🔊 Final Output:\n", final_output)
    if speak_enabled:
        speak(final_output)

    # Optionally save
    if save_output:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Image Path: {image_path}\nScene: {scene}\n")
            f.write(f"Objects: {', '.join(obj_phrases)}\n")
            if region_phrases:
                f.write(f"Regions: {', '.join(region_phrases)}\n")
            if rel_phrases:
                f.write(f"Relations: {', '.join(rel_phrases)}\n")
            f.write(f"Text: {text}\n")
        print(f"💾 Output saved to {output_path}")

    return {
        "scene_description": scene,
        "objects_detected": dict(counts),
        "regions": region_phrases,
        "relationships": rel_phrases,
        "ocr_text": text,
        "full_generated_output": final_output
    }
   


def draw_boxes_on_image(image_path, detections):
    # No drawing needed; display raw image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024,512))
    return ImageTk.PhotoImage(img)


def process_image(image_path, output_text_widget, image_label,speak_enabled=True):
    try:
        scene = describe_scene(image_path)
        detections = detect_objects(image_path)

        # Normalize detections if needed
        if detections and isinstance(detections[0], str):
            detections = [{"label": lbl, "bbox": [0,0,0,0]} for lbl in detections]

        text = read_text_combined(image_path)

        # Count and region logic
        labels = [d["label"] for d in detections]
        counts = Counter(labels)
        img = Image.open(image_path)
        W, H = img.size
        region_dets = {"left": [], "center": [], "right": []}
        for d in detections:
            x1, y1, x2, y2 = d.get("bbox", [0,0,0,0])
            cx = (x1 + x2) / 2
            if cx < W/3:
                region = "left"
            elif cx > 2*W/3:
                region = "right"
            else:
                region = "center"
            region_dets[region].append(d)

        obj_phrases = [f"{cnt} {lbl}{'s' if cnt>1 else ''}" for lbl, cnt in counts.items()]
        region_phrases = []
        for reg, dets in region_dets.items():
            ctr = Counter([d["label"] for d in dets])
            for lbl, cnt in ctr.items():
                region_phrases.append(f"{cnt} {lbl}{'s' if cnt>1 else ''} on the {reg}")

        rel_phrases = []
        for reg, dets in region_dets.items():
            persons = [d for d in dets if d["label"] == "person"]
            vehicles = [d for d in dets if d["label"] in ("motorcycle", "bicycle", "motorbike", "bike")]
            for p in persons:
                px1, py1, px2, py2 = p.get("bbox", [0,0,0,0])
                pcx, pcy = (px1+px2)/2, (py1+py2)/2
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v.get("bbox", [0,0,0,0])
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                        rel_phrases.append(f"a person on a {v['label']} on the {reg}")
                        break

        rel_part = ''
        if rel_phrases:
            rel_part = ' '.join([f"There is {rp}." for rp in rel_phrases]) + ' '

        final_output = (
            f"{rel_part}The image shows {scene}. "
            f"I see {', '.join(obj_phrases)}. "
            f"Specifically, {', '.join(region_phrases)}. "
            #f"The visible text reads: '{text}'."
        )

        # Update GUI
        output_text_widget.delete(1.0, tk.END)
        output_text_widget.insert(tk.END, final_output)
       
        if speak_enabled:
         speak(final_output)

        img_tk = draw_boxes_on_image(image_path, detections)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", str(e))


def launch_gui():
    root = tk.Tk()
    root.title("AI Assist Vision GUI")

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    output_text = tk.Text(root, height=10, width=60, wrap=tk.WORD)
    output_text.pack(pady=10)

    def browse_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            threading.Thread(target=process_image, args=(file_path, output_text, image_label)).start()

    def capture_from_camera():
        file_path = capture_image(use_gui=True)
        if file_path:
            threading.Thread(target=process_image, args=(file_path, output_text, image_label)).start()

    tk.Button(root, text="Select Image", command=browse_image).pack(pady=5)
    tk.Button(root, text="Capture from Camera", command=capture_from_camera).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
