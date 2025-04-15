from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

import cv2


import sys
import os

# Add the main project directory to the Python path
sys.path.append(os.path.abspath("."))  # You can replace "." with your project path

from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak
from modules.camera_capture import capture_image


def run_pipeline(image_path=None, save_output=False, output_path="output.txt"):
    # Capture from webcam if no image path is provided
    if image_path is None:
        image_path = capture_image(use_gui=True)  # Set use_gui=True to press 'c' to capture
                                                  # Set use_gui=False for auto-capture in headless mode

    print("🔍 Analyzing:", image_path)

    # Run modules
    scene = describe_scene(image_path)
    detections = detect_objects(image_path)
    objects = list(set([det["label"] for det in detections]))  # extract just labels
    text = read_text_combined(image_path)

    # Display results
    print("\n🖼️ Scene:", scene)
    print("📦 Objects:", objects)
    print("📝 Text:", text)

   
    # Construct final output
    final_output = (
    f"The image shows {scene}, where objects like {', '.join(objects)} "
    f"are detected, and the visible text reads: '{text}'."
    )

    # Display results
    print("\n🔊 Final Output:\n", final_output)

    # Speak output
    speak(final_output)

    

    # Optionally save output to a file
    if save_output:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Image Path: {image_path}\n")
            f.write(f"Scene: {scene}\n")
            f.write(f"Objects: {', '.join(objects)}\n")
            f.write(f"Text: {text}\n")
        print(f"💾 Output saved to {output_path}")


# -----------------------------------------------
# 📸 Option 1: Use Webcam to capture an image
#     - A window will pop up
#     - Press 'c' to capture
#     - Press 'q' to quit (optional if you extend it)
# -----------------------------------------------
# run_pipeline(image_path=capture_image(use_gui=False), save_output=True)

# Run the pipeline with webcam and interactive capture
# run_pipeline(image_path=None, save_output=True)


# -----------------------------------------------
# 🖼️ Option 2: Use a local image instead
#     - Just uncomment the lines below and specify the path
# -----------------------------------------------

local_image_path = "sample_inputs/finalocr.jpeg"
run_pipeline(image_path=local_image_path, save_output=True)


# for gui

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading

import random

from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak
from modules.camera_capture import capture_image  # For camera feed




def draw_boxes_on_image(image_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Random color per label
    color_map = {}
    for det in detections:
        label = det["label"]
        if label not in color_map:
            color_map[label] = tuple(random.choices(range(50, 256), k=3))

    # Load font (fallback if arial not available)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        label = det["label"]
        confidence = det.get("confidence", 1.0)
        color = color_map[label]

        draw.rectangle(bbox, outline=color, width=2)

        text = f"{label} ({confidence:.2f})"

        # Universal text size calculation
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Older versions fallback
            text_width, text_height = font.getsize(text)

        text_location = (bbox[0], max(bbox[1] - text_height - 4, 0))

        # Draw background rectangle
        draw.rectangle(
            [text_location, (text_location[0] + text_width, text_location[1] + text_height)],
            fill=color
        )

        # Draw text
        draw.text(text_location, text, fill="black", font=font)

    img = img.resize((1024, 512))
    return ImageTk.PhotoImage(img)



def process_image(image_path, output_text_widget, image_label):
    try:
        scene = describe_scene(image_path)
        detections = detect_objects(image_path)

        # If detections are just a list of labels, convert to dict with dummy boxes
        if isinstance(detections[0], str):
            detections = [{"label": label, "bbox": [50, 50, 200, 200]} for label in detections]

        text = read_text_combined(image_path)

        labels = [d["label"] for d in detections]
        final_output = (
            f"The image shows {scene}, where objects like {', '.join(labels)} "
            f"are detected, and the visible text reads: '{text}'."
        )

        # Update text widget
        output_text_widget.delete(1.0, tk.END)
        output_text_widget.insert(tk.END, final_output)

        # Speak it
        speak(final_output)

        # Show image with boxes
        img_tk = draw_boxes_on_image(image_path, detections)
        image_label.configure(image=img_tk)
        image_label.image = img_tk  # Reference to prevent garbage collection

    except Exception as e:
        messagebox.showerror("Error", str(e))

def launch_gui():
    root = tk.Tk()
    root.title("AI Assist Vision GUI")

    # Image display
    image_label = tk.Label(root)
    image_label.pack(pady=10)

    # Output text
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

    # Buttons
    browse_button = tk.Button(root, text="Select Image", command=browse_image)
    browse_button.pack(pady=5)

    camera_button = tk.Button(root, text="Capture from Camera", command=capture_from_camera)
    camera_button.pack(pady=5)

    root.mainloop()

launch_gui()
