from detoxify import Detoxify
from transformers import AutoProcessor, AutoModelForImageClassification , ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# ==== Load models ====
# Load mô hình kiểm duyệt ảnh bạo lực
violence_model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
violence_processor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')

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

# ==== Hàm kiểm duyệt Hình ảnh ====
def is_image_safe(image: Image.Image):
    reasons = []

    # --- Kiểm tra NSFW ---
    nsfw_inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = image_model(**nsfw_inputs)
    nsfw_logits = nsfw_outputs.logits
    nsfw_pred = nsfw_logits.argmax(-1).item()
    nsfw_label = image_model.config.id2label[nsfw_pred]
    if nsfw_label.lower() in ["porn", "hentai", "sexy"]:
        reasons.append(f"Khiêu dâm ({nsfw_label})")

    # --- Kiểm tra Bạo lực ---
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
    violence_logits = violence_outputs.logits
    violence_pred = violence_logits.argmax(-1).item()
    violence_label = violence_model.config.id2label[violence_pred]
    if violence_label.lower() in ["violence", "bloody", "weapon", "fight"]:
        reasons.append(f"Bạo lực ({violence_label})")

    # --- Kết quả ---
    if reasons:
        return False, reasons
    return True, [f"An toàn ({nsfw_label}, {violence_label})"]
