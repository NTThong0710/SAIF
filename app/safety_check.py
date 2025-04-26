from detoxify import Detoxify
from transformers import AutoProcessor, AutoModelForImageClassification, ViTForImageClassification, ViTFeatureExtractor
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


def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    if flagged:
        return False, list(flagged.keys())
    return True, []

def check_image_safe(image: Image.Image):
    # NSFW Check
    nsfw_inputs = nsfw_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = nsfw_model(**nsfw_inputs)
        nsfw_probs = torch.nn.functional.softmax(nsfw_outputs.logits, dim=1)[0]
    nsfw_labels = list(nsfw_model.config.id2label.values())
    nsfw_pred = nsfw_probs.argmax().item()
    nsfw_label = nsfw_labels[nsfw_pred]
    nsfw_score = nsfw_probs[nsfw_pred].item() * 100

        # Violence Check
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
        violence_probs = torch.nn.functional.softmax(violence_outputs.logits, dim=1)[0]
    violence_labels = ["Non-Violent", "Violent"]
    violence_pred = violence_probs.argmax().item()
    violence_label = violence_labels[violence_pred]
    violence_score = violence_probs[violence_pred].item() * 100

    if nsfw_label.lower() in ["porn","hentai","sex","nsfw"]:
        return f"🚨 Ảnh KHÔNG an toàn:\n- Ảnh nhạy cảm ({nsfw_score:.2f}%)"

    if violence_label.lower() in ["Non-Violent"] and violence_score > 50:
        return f"🚨 Ảnh KHÔNG an toàn:\n- Ảnh chứa bạo lực ({violence_score:.2f}%)"
    if violence_label.lower() in ["Violent"]
        return f"🚨 Ảnh KHÔNG an toàn:\n- Ảnh chứa bạo lực ({violence_score:.2f}%)"
    
    return f"✅ Ảnh an toàn\n- NSFW: {nsfw_label} ({nsfw_score:.2f}%)\n- Violence: {violence_label} ({violence_score:.2f}%)"