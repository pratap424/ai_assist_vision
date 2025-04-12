from modules.ocr_reader import read_text_combined
from modules.vlm_captioning import describe_scene
from modules.object_detection import detect_objects
from modules.audio_feedback import speak

def run_pipeline(image_path):
    print("🔍 Analyzing:", image_path)
    
    scene = describe_scene(image_path)
    objects = detect_objects(image_path)
    text = read_text_combined(image_path)

    print("\n🖼️ Scene:", scene)
    print("📦 Objects:", objects)
    print("📝 Text:", text)

    speak(f"I see {scene}. I detected the following objects: {', '.join(objects)}. The text says: {text}")

if __name__ == "__main__":
    run_pipeline("sample_inputs/test_image.jpg")
