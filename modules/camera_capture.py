import cv2
import matplotlib.pyplot as plt

def capture_image(save_path='captured.jpg', camera_index=0, use_gui=True):
    cap = cv2.VideoCapture(camera_index)
    
    # Start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        if use_gui:
            # Convert BGR to RGB for display in matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use matplotlib to display image inline
            plt.imshow(frame_rgb)
            plt.axis('off')  # Hide axes
            plt.show()

            key = cv2.waitKey(1)
            if key == ord('c'):  # Press 'c' to capture
                cv2.imwrite(save_path, frame)
                print(f"Image captured and saved as {save_path}")
                break
            elif key == ord('q'):  # Press 'q' to quit
                print("Capture cancelled.")
                break

        else:
            # Automatically capture without GUI
            cv2.imwrite(save_path, frame)
            break

    cap.release()

    return save_path
