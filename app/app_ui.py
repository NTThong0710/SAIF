import gradio as gr
from app.safety_check import is_prompt_safe, is_image_safe
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt
from PIL import Image

# === Kiểm duyệt prompt ===
def handle_prompt(prompt):
    safe, info = is_prompt_safe(prompt)
    if not safe:
        log_prompt(prompt, info, False, "")
        return f"🚨 Prompt không an toàn! Phát hiện: {', '.join(info)}", ""
    
    response = generate_response(prompt)
    log_prompt(prompt, "OK", True, response)
    return "✅ Prompt an toàn", response

# === Kiểm duyệt ảnh ===
def check_image_safety(image: Image.Image):
    safe, reasons = is_image_safe(image)
    if safe:
        return f"✅ Ảnh an toàn: {', '.join(reasons)}"
    else:
        return f"🚨 Ảnh KHÔNG an toàn: {', '.join(reasons)}"

# === Giao diện ===
with gr.Blocks(title="SAIFGuard") as demo:
    gr.Markdown("## 🛡️ SAIFGuard")
    
    with gr.Tab("📝 Kiểm duyệt Prompt"):
        prompt_input = gr.Textbox(label="Nhập Prompt")
        prompt_status = gr.Textbox(label="Trạng thái kiểm duyệt")
        prompt_output = gr.Textbox(label="Kết quả GenAI")
        prompt_button = gr.Button("Kiểm tra Prompt")
        prompt_button.click(handle_prompt, inputs=prompt_input, outputs=[prompt_status, prompt_output])

    with gr.Tab("🖼️ Kiểm duyệt Hình ảnh"):
        image_input = gr.Image(type="pil", label="Tải ảnh lên")
        image_output = gr.Textbox(label="Trạng thái kiểm duyệt hình ảnh")
        image_button = gr.Button("Kiểm tra Hình ảnh")
        image_button.click(fn=check_image_safety, inputs=image_input, outputs=image_output)

demo.launch()
