import cv2
from PIL import Image, ImageTk
import tkinter as tk
from ultralytics import YOLO
import threading
import time
import pyttsx3  # 👈 New import

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

class LiveObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 Live Object Detection with Voice")

        self.label = tk.Label(root)
        self.label.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.last_objects = []

        # TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # Start narration thread
        self.narration_thread = threading.Thread(target=self.narrate_loop, daemon=True)
        self.narration_thread.start()

        self.update_frame()

    def detect_objects(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(rgb, verbose=False)[0]
        names = results.names

        current_objects = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            current_objects.append(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.last_objects = list(set(current_objects))  # 👈 Save for narration
        return frame

    def narrate_loop(self):
        while self.running:
            if self.last_objects:
                objects_str = ", ".join(self.last_objects)
                speech = f"I see {objects_str}."
                self.tts_engine.say(speech)
                self.tts_engine.runAndWait()
            time.sleep(5)  # 👈 Narrate every 5 seconds

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = self.detect_objects(frame)
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveObjectDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
