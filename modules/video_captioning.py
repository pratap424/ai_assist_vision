import cv2
import torch
import numpy as np
from PIL import Image
import os
import tempfile
from collections import Counter
from datetime import datetime
import time

# Import existing modules for consistent results
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects

class VideoCaptioningProcessor:
    def __init__(self):
        """Initialize the video captioning processor"""
        print("📚 Loading video captioning module...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Using existing models loaded in other modules
        print(f"✅ Video captioning module ready on {self.device}")
    
    def extract_frames(self, video_path, n_frames=5, method="uniform"):
        """
        Extract representative frames from a video
        
        Args:
            video_path (str): Path to the video file
            n_frames (int): Number of frames to extract
            method (str): Method to use for sampling frames ('uniform', 'keyframe', etc.)
            
        Returns:
            list: List of extracted frame paths
        """
        print(f"🎬 Extracting {n_frames} frames from video...")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        if method == "uniform" and total_frames > 0:
            # Uniform sampling - evenly distributed frames
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
            
            for i, frame_idx in enumerate(frame_indices):
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
        
        elif method == "keyframe":
            # Simple keyframe extraction based on differences between frames
            prev_frame = None
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or saved_count >= n_frames:
                    break
                
                if frame_count % (total_frames // (n_frames * 5)) == 0:
                    # Convert to grayscale for comparison
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Compare with previous frame if available
                    if prev_frame is not None:
                        diff = cv2.absdiff(gray, prev_frame)
                        non_zero_count = np.count_nonzero(diff)
                        
                        # If significant difference, save as keyframe
                        if non_zero_count > gray.size * 0.1:  # 10% change threshold
                            frame_path = os.path.join(temp_dir, f"frame_{saved_count:03d}.jpg")
                            cv2.imwrite(frame_path, frame)
                            frame_paths.append(frame_path)
                            saved_count += 1
                    else:
                        # Always save first frame
                        frame_path = os.path.join(temp_dir, f"frame_{saved_count:03d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        saved_count += 1
                    
                    prev_frame = gray
                
                frame_count += 1
        
        # Release video capture
        cap.release()
        
        # If we couldn't extract enough frames, add some randomly
        if len(frame_paths) < n_frames:
            print(f"⚠️ Could only extract {len(frame_paths)} frames, adding random frames...")
            
            cap = cv2.VideoCapture(video_path)
            while len(frame_paths) < n_frames:
                random_idx = np.random.randint(0, total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = os.path.join(temp_dir, f"frame_r{len(frame_paths):03d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
            
            cap.release()
        
        print(f"✅ Extracted {len(frame_paths)} frames")
        return frame_paths
    
    def analyze_frames(self, frame_paths):
        """
        Analyze a set of frames from a video
        
        Args:
            frame_paths (list): List of frame image paths
            
        Returns:
            dict: Analysis data for the frames
        """
        print("🔍 Analyzing video frames...")
        
        # Collect scene descriptions for each frame
        scene_descriptions = []
        all_objects = []
        
        for frame_path in frame_paths:
            # Get scene description using existing module
            description = describe_scene(frame_path)
            scene_descriptions.append(description)
            
            # Detect objects using existing module
            detections = detect_objects(frame_path)
            frame_objects = [d["label"] for d in detections]
            all_objects.extend(frame_objects)
        
        # Count objects across all frames
        object_counter = Counter(all_objects)
        top_objects = object_counter.most_common(5)
        
        # Analyze scene descriptions for common themes
        all_words = ' '.join(scene_descriptions).lower().split()
        word_counter = Counter(all_words)
        common_words = [word for word, count in word_counter.most_common(10) 
                        if len(word) > 3 and word not in ['this', 'that', 'with', 'from']]
        
        return {
            "scene_descriptions": scene_descriptions,
            "top_objects": top_objects,
            "common_themes": common_words,
            "detection_counts": dict(object_counter)
        }
    
    def generate_video_description(self, analysis_data):
        """
        Generate a natural language description of a video based on frame analysis
        
        Args:
            analysis_data (dict): Analysis data from analyze_frames
            
        Returns:
            str: Natural language description of the video
        """
        print("✍️ Generating video description...")
        
        descriptions = analysis_data["scene_descriptions"]
        top_objects = analysis_data["top_objects"]
        common_themes = analysis_data["common_themes"]
        
        # Combine the most representative scene descriptions
        # Using the first, a middle, and the last description
        representative_scenes = [
            descriptions[0],  # First frame
            descriptions[len(descriptions) // 2],  # Middle frame
            descriptions[-1]  # Last frame
        ]
        
        # Construct natural language description
        description_parts = []
        
        # Overall content
        if top_objects:
            objects_text = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" 
                                    for obj, count in top_objects])
            description_parts.append(f"This video contains {objects_text}.")
        
        # Scene progression
        if len(representative_scenes) >= 3:
            description_parts.append(f"The video begins with {representative_scenes[0].lower()}.")
            description_parts.append(f"Then, {representative_scenes[1].lower()}.")
            description_parts.append(f"Finally, {representative_scenes[-1].lower()}.")
        
        # Common themes if relevant
        if common_themes:
            themes = ", ".join(common_themes[:3])
            description_parts.append(f"The main themes in this video are related to {themes}.")
        
        # Join parts into a coherent description
        full_description = " ".join(description_parts)
        
        print(f"✅ Description generated: {full_description}")
        return full_description
    
    def caption_video(self, video_path, n_frames=5):
        """
        Generate a comprehensive caption for a video
        
        Args:
            video_path (str): Path to the video file
            n_frames (int): Number of frames to analyze
            
        Returns:
            str: Natural language caption of the video
            dict: Analysis data
        """
        start_time = time.time()
        print(f"🎥 Captioning video: {video_path}")
        
        # Extract frames
        frame_paths = self.extract_frames(video_path, n_frames=n_frames)
        
        # Analyze frames
        analysis_data = self.analyze_frames(frame_paths)
        
        # Generate description
        description = self.generate_video_description(analysis_data)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Video captioning completed in {elapsed_time:.2f} seconds")
        
        # Clean up temporary frame files (optional)
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        
        return description, analysis_data
    
    def record_and_caption_video(self, duration=10, camera_id=0):
        """
        Record a video and generate a caption for it
        
        Args:
            duration (int): Duration to record in seconds
            camera_id (int): Camera index to use
            
        Returns:
            str: Path to saved video
            str: Caption for the video
        """
        # Create output directory if it doesn't exist
        os.makedirs("output/videos", exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return None, "Failed to open camera"
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"output/videos/video_{timestamp}.mp4"
        
        # Create video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        print(f"🎥 Recording video for {duration} seconds...")
        
        # Record for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame to video
            out.write(frame)
            
            # Display recording status
            cv2.putText(frame, f"Recording: {int(time.time() - start_time)}s / {duration}s", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Recording...", frame)
            
            # Check for ESC key to cancel
            if cv2.waitKey(1) == 27:
                break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"💾 Video saved to {video_path}")
        
        # Caption the recorded video
        caption, _ = self.caption_video(video_path)
        
        return video_path, caption