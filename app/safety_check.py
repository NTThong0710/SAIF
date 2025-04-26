from detoxify import Detoxify
from transformers import (
    AutoProcessor, AutoModelForImageClassification,
    ViTForImageClassification, ViTFeatureExtractor,
    BlipProcessor, BlipForConditionalGeneration
)
from PIL import Image
import torch

# Load models
detox_model = Detoxify('original')

nsfw_model_id = "Falconsai/nsfw_image_detection"
nsfw_processor = AutoProcessor.from_pretrained(nsfw_model_id)
nsfw_model = AutoModelForImageClassification.from_pretrained(nsfw_model_id)

violence_model_id = "jaranohaal/vit-base-violence-detection"
violence_model = ViTForImageClassification.from_pretrained(violence_model_id)
violence_processor = ViTFeatureExtractor.from_pretrained(violence_model_id)

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    if flagged:
        return False, list(flagged.keys())
    return True, []

def generate_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def check_nsfw_image(image: Image.Image) -> str:
    """Kiểm tra và trả về kết quả NSFW của ảnh"""
    # Xử lý NSFW
    nsfw_inputs = nsfw_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = nsfw_model(**nsfw_inputs)
        nsfw_probs = torch.nn.functional.softmax(nsfw_outputs.logits, dim=1)[0]
    
    nsfw_labels = list(nsfw_model.config.id2label.values())
    nsfw_pred = nsfw_probs.argmax().item()
    nsfw_label = nsfw_labels[nsfw_pred]
    nsfw_score = nsfw_probs[nsfw_pred].item() * 100
    
    # Tạo caption
    caption = generate_caption(image)
    
    # Đánh giá kết quả
    if nsfw_label.lower() in ["porn", "hentai", "sex", "nsfw"]:
        return f"""🚨 Ảnh KHÔNG an toàn (NSFW):
- Loại: {nsfw_label}
- Độ chính xác: {nsfw_score:.2f}%
- Mô tả: {caption}"""
    else:
        return f"""✅ Ảnh an toàn (NSFW):
- Loại: {nsfw_label}
- Độ chính xác: {nsfw_score:.2f}%
- Mô tả: {caption}"""

def check_violence_image(image: Image.Image) -> str:
    """Kiểm tra và trả về kết quả bạo lực của ảnh"""
    # Xử lý bạo lực
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
        violence_probs = torch.nn.functional.softmax(violence_outputs.logits, dim=1)[0]
    
    violence_labels = ["Non-Violent", "Violent"]
    violence_pred = violence_probs.argmax().item()
    violence_label = violence_labels[violence_pred]
    violence_score = violence_probs[violence_pred].item() * 100
    
    # Tạo caption
    caption = generate_caption(image)
    
    # Đánh giá kết quả
    is_violent = False
    if violence_label.lower() == "non-violent" and violence_score > 50:
        is_violent = True
    elif violence_label.lower() == "violent" and violence_score > 80:
        is_violent = True
    
    if is_violent:
        return f"""🚨 Ảnh KHÔNG an toàn (Bạo lực):
- Loại: {violence_label}
- Độ chính xác: {violence_score:.2f}%
- Mô tả: {caption}"""
    else:
        return f"""✅ Ảnh an toàn (Bạo lực):
- Loại: {violence_label}
- Độ chính xác: {violence_score:.2f}%
- Mô tả: {caption}"""
