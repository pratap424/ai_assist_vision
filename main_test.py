import os
import sys
import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
import argparse

# Set matplotlib to non-GUI backend if running on Render
if os.environ.get("RENDER", "0") == "1":
    import matplotlib
    matplotlib.use('Agg')

# Add the main project directory to the Python path
sys.path.append(os.path.abspath("."))

# Check if we're running in Jupyter and handle argparse accordingly
in_jupyter = 'ipykernel' in sys.modules

# Import modules
from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak
from modules.camera_capture import capture_image
from modules.vqa_module import VQAProcessor  # New VQA module
from modules.video_captioning import VideoCaptioningProcessor  # New video captioning module


def run_pipeline(image_path=None, save_output=False, output_path="output.txt", speak_enabled=True):
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

    # Determine spatial regions
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

    # Build output phrases
    obj_phrases = [f"{cnt} {lbl}{'s' if cnt>1 else ''}" for lbl, cnt in counts.items()]
    region_phrases = []
    for reg, dets in region_dets.items():
        ctr = Counter([d["label"] for d in dets])
        for lbl, cnt in ctr.items():
            region_phrases.append(f"{cnt} {lbl}{'s' if cnt>1 else ''} on the {reg}")

    # Detect relationships
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

    # Final output construction
    rel_part = ''
    if rel_phrases:
        rel_part = ' '.join([f"There is {rp}." for rp in rel_phrases]) + ' '
    final_output = (
        f"{rel_part}The image shows {scene}. "
        f"I see {', '.join(obj_phrases)}. "
        f"Specifically, {', '.join(region_phrases)}."
    )

    print("\n🔊 Final Output:\n", final_output)

    if speak_enabled:
        speak(final_output)

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


# Enhanced version with query support and video captioning
def answer_image_query(image_path, query, speak_enabled=True):
    """Process a natural language query about an image"""
    # Initialize VQA processor if not already initialized
    global vqa_processor
    if 'vqa_processor' not in globals():
        vqa_processor = VQAProcessor()
    
    # First run standard pipeline to get image data
    image_data = run_pipeline(image_path, save_output=False, speak_enabled=False)
    
    # Now answer the specific query
    answer = vqa_processor.answer_query_with_context(image_path, query, image_data)
    
    print(f"\n❓ Query: {query}")
    print(f"💬 Answer: {answer}")
    
    if speak_enabled:
        speak(answer)
    
    return answer


def caption_video(video_path=None, duration=10, camera_id=0, speak_enabled=True):
    """Record or use an existing video and generate a caption"""
    # Initialize video captioning processor if not already initialized
    global video_processor
    if 'video_processor' not in globals():
        video_processor = VideoCaptioningProcessor()
    
    # Either use provided video or record a new one
    if video_path is None:
        video_path, caption = video_processor.record_and_caption_video(duration, camera_id)
    else:
        caption, _ = video_processor.caption_video(video_path)
    
    print(f"\n🎬 Video: {video_path}")
    print(f"📝 Caption: {caption}")
    
    if speak_enabled:
        speak(caption)
    
    return caption


# GUI part: only imports GUI libraries inside function
def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
    import threading
    import cv2
    
    # Track current loaded image and video paths
    current_image_path = None
    current_video_path = None
    
    def draw_boxes_on_image(image_path, detections):
        img = Image.open(image_path).convert("RGB")
        # Scale down large images for display
        max_width = 800
        if img.width > max_width:
            scale = max_width / img.width
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height))
        return ImageTk.PhotoImage(img)

    def process_image(image_path, output_text_widget, image_label, speak_enabled=True):
        try:
            scene = describe_scene(image_path)
            detections = detect_objects(image_path)

            if detections and isinstance(detections[0], str):
                detections = [{"label": lbl, "bbox": [0,0,0,0]} for lbl in detections]

            text = read_text_combined(image_path)

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
                f"Specifically, {', '.join(region_phrases)}."
            )

            output_text_widget.delete(1.0, tk.END)
            output_text_widget.insert(tk.END, final_output)

            if speak_enabled:
                speak(final_output)

            img_tk = draw_boxes_on_image(image_path, detections)
            image_label.configure(image=img_tk)
            image_label.image = img_tk
            
            # Store the current image path for query mode
            nonlocal current_image_path
            current_image_path = image_path

        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def process_query(query_var, output_text, image_label, speak_var):
        """Process a query about the current image"""
        query = query_var.get()
        if not query:
            messagebox.showinfo("Input Needed", "Please enter a question about the image.")
            return
            
        nonlocal current_image_path
        if not current_image_path:
            messagebox.showinfo("No Image", "Please load an image first.")
            return
            
        try:
            answer = answer_image_query(current_image_path, query, speak_var.get())
            
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"Q: {query}\nA: {answer}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def process_video(video_path, output_text, video_duration_var, camera_id_var, speak_var):
        """Process a video file or record from camera"""
        try:
            nonlocal current_video_path
            
            # If no path provided, record from camera
            if not video_path:
                duration = int(video_duration_var.get())
                camera_id = int(camera_id_var.get())
                caption = caption_video(None, duration, camera_id, speak_var.get())
            else:
                caption = caption_video(video_path, speak_enabled=speak_var.get())
                current_video_path = video_path
            
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"Video: {current_video_path}\n\nCaption: {caption}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def browse_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            # Start processing in a separate thread to keep UI responsive
            threading.Thread(
                target=process_image, 
                args=(file_path, output_text, image_display, speech_var.get()),
                daemon=True
            ).start()
    
    def capture_from_camera():
        try:
            # Capture image using the camera module
            img_path = capture_image(use_gui=False)
            if img_path:
                threading.Thread(
                    target=process_image, 
                    args=(img_path, output_text, image_display, speech_var.get()),
                    daemon=True
                ).start()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
    
    def browse_video():
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            threading.Thread(
                target=process_video,
                args=(file_path, output_text, video_duration_var, camera_id_var, speech_var),
                daemon=True
            ).start()
    
    def record_video():
        threading.Thread(
            target=process_video,
            args=(None, output_text, video_duration_var, camera_id_var, speech_var),
            daemon=True
        ).start()
    
    def submit_query():
        threading.Thread(
            target=process_query,
            args=(query_var, output_text, image_display, speech_var),
            daemon=True
        ).start()
    
    def save_results():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(output_text.get(1.0, tk.END))
            messagebox.showinfo("Save", f"Results saved to {file_path}")
    
    # Create main window
    root = tk.Tk()
    root.title("Vision Assistant")
    root.geometry("1200x800")
    
    # Create tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Image tab
    image_tab = ttk.Frame(notebook)
    notebook.add(image_tab, text="Image Analysis")
    
    # Video tab
    video_tab = ttk.Frame(notebook)
    notebook.add(video_tab, text="Video Captioning")
    
    # Query tab
    query_tab = ttk.Frame(notebook)
    notebook.add(query_tab, text="Ask Questions")
    
    # Common elements - output text and image display
    output_frame = ttk.Frame(root)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Split the output area
    output_paned = ttk.PanedWindow(output_frame, orient=tk.HORIZONTAL)
    output_paned.pack(fill=tk.BOTH, expand=True)
    
    # Left side - image display
    image_frame = ttk.Frame(output_paned)
    output_paned.add(image_frame, weight=3)
    
    # Image display label
    image_display = ttk.Label(image_frame)
    image_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Right side - text output
    text_frame = ttk.Frame(output_paned)
    output_paned.add(text_frame, weight=2)
    
    # Text display with scrollbar
    output_text = tk.Text(text_frame, wrap=tk.WORD, height=20)
    output_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)
    
    scrollbar = ttk.Scrollbar(text_frame, command=output_text.yview)
    scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
    output_text.config(yscrollcommand=scrollbar.set)
    
    # Bottom controls frame
    controls_frame = ttk.Frame(root)
    controls_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Speech synthesis toggle
    speech_var = tk.BooleanVar(value=True)
    speech_check = ttk.Checkbutton(controls_frame, text="Enable Speech", variable=speech_var)
    speech_check.pack(side=tk.LEFT, padx=5)
    
    # Save results button
    save_button = ttk.Button(controls_frame, text="Save Results", command=save_results)
    save_button.pack(side=tk.RIGHT, padx=5)
    
    # === IMAGE TAB CONTROLS ===
    image_controls = ttk.Frame(image_tab)
    image_controls.pack(fill=tk.X, padx=10, pady=10)
    
    browse_img_btn = ttk.Button(image_controls, text="Load Image", command=browse_image)
    browse_img_btn.pack(side=tk.LEFT, padx=5)
    
    camera_btn = ttk.Button(image_controls, text="Capture from Camera", command=capture_from_camera)
    camera_btn.pack(side=tk.LEFT, padx=5)
    
    # === VIDEO TAB CONTROLS ===
    video_controls = ttk.Frame(video_tab)
    video_controls.pack(fill=tk.X, padx=10, pady=10)
    
    browse_video_btn = ttk.Button(video_controls, text="Load Video", command=browse_video)
    browse_video_btn.pack(side=tk.LEFT, padx=5)
    
    record_video_btn = ttk.Button(video_controls, text="Record Video", command=record_video)
    record_video_btn.pack(side=tk.LEFT, padx=5)
    
    # Video duration and camera ID
    video_settings = ttk.LabelFrame(video_tab, text="Settings")
    video_settings.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(video_settings, text="Duration (s):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    video_duration_var = tk.StringVar(value="10")
    ttk.Entry(video_settings, textvariable=video_duration_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")
    
    ttk.Label(video_settings, text="Camera ID:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
    camera_id_var = tk.StringVar(value="0")
    ttk.Entry(video_settings, textvariable=camera_id_var, width=5).grid(row=0, column=3, padx=5, pady=5, sticky="w")
    
    # === QUERY TAB CONTROLS ===
    query_controls = ttk.Frame(query_tab)
    query_controls.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Label(query_controls, text="Ask about the image:").pack(side=tk.LEFT, padx=5)
    
    query_var = tk.StringVar()
    query_entry = ttk.Entry(query_controls, textvariable=query_var, width=50)
    query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    query_btn = ttk.Button(query_controls, text="Ask", command=submit_query)
    query_btn.pack(side=tk.LEFT, padx=5)
    
    # Load image button for query tab
    load_for_query_btn = ttk.Button(query_tab, text="Load Image for Query", command=browse_image)
    load_for_query_btn.pack(padx=10, pady=5)
    
    # Start with a welcome message
    output_text.insert(tk.END, "Welcome to Vision Assistant!\n\n"
                      "Use the tabs above to:\n"
                      "1. Analyze images\n"
                      "2. Caption videos\n"
                      "3. Ask questions about images\n\n"
                      "Get started by loading an image or recording a video.")
    
    # Start the GUI main loop
    root.mainloop()


if __name__ == "__main__":
    # Check if we're running in Jupyter
    if in_jupyter:
        # For Jupyter, provide a direct function call instead of using argparse
        def run_in_jupyter(mode='gui', image_path=None, video_path=None, query=None, save_output=False, output_file="output.txt", speak_enabled=True):
            """
            Run the Vision Assistant in Jupyter notebook
            
            Parameters:
            -----------
            mode : str
                'gui' - Launch the GUI interface
                'image' - Process a single image
                'video' - Process a video
                'query' - Ask a question about an image
            image_path : str
                Path to an image file (required for 'image' and 'query' modes)
            video_path : str
                Path to a video file (required for 'video' mode, or None to record)
            query : str
                Question to ask about the image (for 'query' mode)
            save_output : bool
                Whether to save the output to a file
            output_file : str
                Path to save the output (if save_output is True)
            speak_enabled : bool
                Whether to enable speech output
            """
            if mode == 'gui':
                launch_gui()
            elif mode == 'image' and image_path:
                run_pipeline(image_path, save_output, output_file, speak_enabled)
            elif mode == 'video':
                caption_video(video_path, speak_enabled=speak_enabled)
            elif mode == 'query' and image_path and query:
                answer_image_query(image_path, query, speak_enabled)
            else:
                print("Invalid arguments for run_in_jupyter()")
                print("Example usage:")
                print("- run_in_jupyter('gui')  # Launch GUI")
                print("- run_in_jupyter('image', 'path/to/image.jpg')  # Process image")
                print("- run_in_jupyter('video', video_path='path/to/video.mp4')  # Process video")
                print("- run_in_jupyter('query', 'path/to/image.jpg', query='What objects are in this image?')  # Ask question")
                
        # Make the function available when imported in Jupyter
        globals()['run_in_jupyter'] = run_in_jupyter
        
        # By default, just run the GUI when the file is executed
        print("Running in Jupyter environment. Use run_in_jupyter() function to process images/videos.")
        # Don't auto-launch GUI to avoid blocking the notebook
        
    else:
        # Normal command-line execution
        try:
            parser = argparse.ArgumentParser(description="Vision Assistant")
            parser.add_argument("--image", type=str, help="Path to image file")
            parser.add_argument("--video", type=str, help="Path to video file")
            parser.add_argument("--save", action="store_true", help="Save output to file")
            parser.add_argument("--output", type=str, default="output.txt", help="Output file path")
            parser.add_argument("--no-speak", action="store_true", help="Disable speech output")
            parser.add_argument("--gui", action="store_true", help="Launch GUI")
            parser.add_argument("--record", type=int, default=0, help="Record video for N seconds")
            parser.add_argument("--query", type=str, help="Ask a question about the image")
            
            args = parser.parse_args()
            
            # Launch GUI if requested or if no arguments provided
            if args.gui or (len(sys.argv) == 1):
                launch_gui()
            else:
                # Command-line mode
                speak_enabled = not args.no_speak
                
                if args.video:
                    # Process video
                    caption_video(args.video, speak_enabled=speak_enabled)
                elif args.record > 0:
                    # Record and process video
                    caption_video(None, args.record, speak_enabled=speak_enabled)
                elif args.image:
                    # Process image
                    if args.query:
                        # Answer query about image
                        answer_image_query(args.image, args.query, speak_enabled=speak_enabled)
                    else:
                        # Basic image analysis
                        run_pipeline(args.image, args.save, args.output, speak_enabled=speak_enabled)
                else:
                    # No image or video specified, show help
                    parser.print_help()
        except SystemExit:
            # Catch SystemExit to avoid problems in environments that don't handle it well
            pass



# Run the GUI directly
run_in_jupyter('gui')