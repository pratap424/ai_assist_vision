from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import easyocr

# Load models once
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def read_text_combined(image_path):
    print("\n🔍 Performing OCR with TrOCR and EasyOCR...")

    image = preprocess_image(image_path)
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    result_easyocr = reader.readtext(image_path)
    easy_text = [item[1] for item in result_easyocr if len(item[1]) > 1]
    easyocr_combined = " ".join(easy_text)

    final_text = trocr_text.strip() + ". " + easyocr_combined.strip()
    return final_text if final_text.strip() else "No readable text found."
