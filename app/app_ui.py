import gradio as gr
from app.safety_check import is_prompt_safe
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt

# === Nhận diện ảnh nhạy cảm ===
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Tải model NSFW detector (có thể thay đổi model nếu muốn)
model_id = "Falconsai/nsfw_image_detection"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

def check_image_nsfw(image: Image.Image):
    labels = model.config.id2label.values()  # lấy nhãn từ config: safe, nsfw
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    safe_prob = probs[0].item()
    nsfw_prob = probs[1].item()

    if nsfw_prob > safe_prob:
        return f"🚨 Ảnh KHÔNG an toàn ({nsfw_prob * 100:.2f}%)"
    else:
        return f"✅ Ảnh an toàn ({safe_prob * 100:.2f}%)"

# === Kiểm duyệt prompt ===
def handle_prompt(prompt):
    log_prompt(prompt, "OK", True, response)
    return "✅ Prompt an toàn", response
    
# === Giao diện ===
with gr.Blocks(title="SAIFGuard - GENAI HỖ TRỢ PHÁT HIỆN PROMPT & IMAGE KHÔNG AN TOÀN", css="""
.yellow-btn {
    background-color: #FFD700 !important;
    color: black !important;
}
""") as demo:
    gr.Markdown("## 🛡️ SAIFGuard: GenAI Prompt & Image Safety Checker")
    
    with gr.Tab("📝 Kiểm duyệt Prompt"):
        with gr.Row():
            # Cột bên trái cho status, output và button
            with gr.Column(scale=1):
                prompt_status = gr.Textbox(label="Trạng thái kiểm duyệt")
                prompt_output = gr.Textbox(label="Kết quả GenAI")
                prompt_button = gr.Button("Kiểm tra Prompt", elem_classes="yellow-btn")
            
            # Cột bên phải cho input
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="Nhập Prompt", lines=10)
        
        prompt_button.click(handle_prompt, inputs=prompt_input, outputs=[prompt_status, prompt_output])
    
    with gr.Tab("🖼️ Kiểm duyệt Hình ảnh"):
        with gr.Row():
            # Cột bên trái cho output và button
            with gr.Column(scale=1):
                image_output = gr.Textbox(label="Trạng thái kiểm duyệt hình ảnh")
                image_button = gr.Button("Kiểm tra Hình ảnh", elem_classes="yellow-btn")
            
            # Cột bên phải cho image input
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Tải ảnh lên")
                
        image_button.click(fn=check_image_nsfw, inputs=image_input, outputs=image_output)