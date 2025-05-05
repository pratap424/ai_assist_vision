from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_scene(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)
    
def answer_query(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, text=question, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=50)
    answer = blip_processor.decode(output[0], skip_special_tokens=True)
    return answer
