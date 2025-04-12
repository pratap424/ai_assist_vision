from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_scene(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)
