from detoxify import Detoxify
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# ==== Load models ====
# Load mô hình kiểm duyệt văn bản
detox_model = Detoxify('original')

# Load mô hình kiểm duyệt ảnh
image_processor = AutoProcessor.from_pretrained("Falconsai/nsfw_image_detection")
image_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# ==== Hàm kiểm duyệt prompt ====
def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    
    # Nếu bất kỳ chỉ số độc hại nào > 0.5 thì xem là không an toàn
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    
    if flagged:
        return False, list(flagged.keys())
    return True, []

# ==== Hàm kiểm duyệt hình ảnh ====
def is_image_safe(image: Image.Image):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)

    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    label = image_model.config.id2label[predicted_class]

    if label.lower() in ["porn", "hentai", "sexy"]:
        return False, label
    return True, label