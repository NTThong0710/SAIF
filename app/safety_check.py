from detoxify import Detoxify
from transformers import AutoProcessor, AutoModelForImageClassification, ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# ==== Load models ====
# Load model kiểm duyệt văn bản độc hại
detox_model = Detoxify('original')

# Load model ảnh nhạy cảm (NSFW)
nsfw_processor = AutoProcessor.from_pretrained("Falconsai/nsfw_image_detection")
nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# Load model bạo lực (Violence Detection)
violence_model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
violence_processor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')


# ==== Kiểm duyệt prompt ====
def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    if flagged:
        return False, list(flagged.keys())
    return True, []


# ==== Kiểm duyệt hình ảnh ====
def is_image_safe(image: Image.Image, violence_threshold=0.5):
    reasons = []
    # --- NSFW detection ---
    nsfw_inputs = nsfw_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = nsfw_model(**nsfw_inputs)
    nsfw_label = nsfw_model.config.id2label[nsfw_outputs.logits.argmax(-1).item()].lower()
    if nsfw_label in ["porn", "hentai", "sexy"]:
        reasons.append("nhạy cảm")

    # --- Violence detection ---
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
    violence_label_index = violence_outputs.logits.argmax(-1).item()
    violence_score = torch.nn.functional.softmax(violence_outputs.logits, dim=-1)[0][violence_label_index].item()

    if violence_label_index == violence_model.config.label2id["violent"] and violence_score > violence_threshold:
        reasons.append(f"bạo lực (confidence: {violence_score:.2f})")

    if reasons:
        return False, f"❌ Không an toàn: {', '.join(reasons)}"
    else:
        return True, "✅ Hình ảnh an toàn"

