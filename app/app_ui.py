import gradio as gr
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt

# === Kiểm duyệt prompt ===
from detoxify import Detoxify

# === Kiểm duyệt ảnh: NSFW + Violence ===
from transformers import AutoProcessor, AutoModelForImageClassification, ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Load NSFW detector
nsfw_model_id = "Falconsai/nsfw_image_detection"
nsfw_processor = AutoProcessor.from_pretrained(nsfw_model_id)
nsfw_model = AutoModelForImageClassification.from_pretrained(nsfw_model_id)

# Load Violence detector
violence_model_id = "jaranohaal/vit-base-violence-detection"
violence_model = ViTForImageClassification.from_pretrained(violence_model_id)
violence_processor = ViTFeatureExtractor.from_pretrained(violence_model_id)

# ==== Kiểm duyệt hình ảnh (image) ====
def check_image_safe(image: Image.Image):
    reasons = []
    
    # === NSFW Check ===
    nsfw_inputs = nsfw_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = nsfw_model(**nsfw_inputs)
        nsfw_probs = torch.nn.functional.softmax(nsfw_outputs.logits, dim=1)[0]

    nsfw_labels = list(nsfw_model.config.id2label.values())
    nsfw_pred = nsfw_probs.argmax().item()
    nsfw_label = nsfw_labels[nsfw_pred]

    if nsfw_label.lower() in ["porn", "hentai", "sexy"] and nsfw_score > 55:
        reasons.append(f"Ảnh nhạy cảm ({nsfw_score:.2f}% - {nsfw_label})")

    # === Violence Check ===
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
        violence_probs = torch.nn.functional.softmax(violence_outputs.logits, dim=1)[0]

    violence_labels = list(violence_model.config.id2label.values())
    violence_confidences = {label: violence_probs[i].item() for i, label in enumerate(violence_labels)}
    violence_pred = violence_probs.argmax().item()
    violence_label = violence_labels[violence_pred]
    violence_score = violence_confidences[violence_label] * 100  # Convert to percentage

    # Kiểm tra Violence (sử dụng label chính xác từ model)
    if violence_label.lower() in ["violent", "violence", "label_1"] and violence_score > 50:
        reasons.append(f"Ảnh chứa bạo lực ({violence_score:.2f}%)")

    # === Tổng kết ===
    if reasons:
        return f"🚨 Ảnh KHÔNG an toàn:\n- " + "\n- ".join(reasons)
    else:
        return f"✅ Ảnh an toàn\n- NSFW: {nsfw_label} ({nsfw_score:.2f}%)\n- Violence: {violence_label} ({violence_score:.2f}%)"

# === Kiểm duyệt Prompt ===
def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    if flagged:
        return False, list(flagged.keys())
    return True, []

# === Giao diện ===
with gr.Blocks(title="SAIFGuard - HỆ THỐNG KIỂM DUYỆT THÔNG MINH", css="""
.yellow-btn {
    background-color: #FFD700 !important;
    color: black !important;
}
""") as demo:
    gr.Markdown("## 🛡️ SAIFGuard: HỆ THỐNG KIỂM DUYỆT THÔNG MINH")
    
    with gr.Tab("📝 Kiểm duyệt Prompt"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="Nhập Prompt", lines=2)
            with gr.Column(scale=1):
                prompt_status = gr.Textbox(label="Trạng thái kiểm duyệt")
                prompt_output = gr.Textbox(label="Kết quả GenAI")
                prompt_button = gr.Button("Kiểm tra Prompt", elem_classes="yellow-btn")
        prompt_button.click(handle_prompt, inputs=prompt_input, outputs=[prompt_status, prompt_output])
    
    with gr.Tab("🖼️ Kiểm duyệt Hình ảnh"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Tải ảnh lên")
            with gr.Column(scale=1):
                image_output = gr.Textbox(label="Trạng thái kiểm duyệt hình ảnh")
                image_button = gr.Button("Kiểm tra Hình ảnh", elem_classes="yellow-btn")
        image_button.click(fn=check_image_safe, inputs=image_input, outputs=image_output)
