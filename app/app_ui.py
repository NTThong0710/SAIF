import gradio as gr
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt
from app.safety_check import check_image_safe, is_prompt_safe

# === Kiểm duyệt Prompt ===
def handle_prompt(prompt):
    is_safe, reasons = is_prompt_safe(prompt)
    if not is_safe:
        return f"❌ Prompt không an toàn: {', '.join(reasons)}", ""
    else:
        log_prompt(prompt)
        response = generate_response(prompt)
        return "✅ Prompt an toàn", response

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
