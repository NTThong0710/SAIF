import gradio as gr
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt
from app.safety_check import check_nsfw_image, check_violence_image, is_prompt_safe, check_url

# === Kiểm duyệt Prompt ===
def handle_prompt(prompt):
    safe, info = is_prompt_safe(prompt)
    if not safe:
        log_prompt(prompt, info, False, "")
        return f"🚨 Prompt không an toàn! Phát hiện: {', '.join(info)}", ""
    
    response = generate_response(prompt)
    log_prompt(prompt, "OK", True, response)
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
        gr.Markdown("### 📷 Tải ảnh và kiểm tra từng tiêu chí")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Tải ảnh lên")

        with gr.Row():
            with gr.Column(scale=1):
                nsfw_output = gr.Textbox(label="🔞 Kết quả kiểm duyệt ảnh nhạy cảm")
                nsfw_button = gr.Button("Kiểm tra Ảnh Nhạy Cảm", elem_classes="yellow-btn")
            with gr.Column(scale=1):
                violence_output = gr.Textbox(label="🧨 Kết quả kiểm duyệt ảnh bạo lực")
                violence_button = gr.Button("Kiểm tra Ảnh Bạo Lực", elem_classes="yellow-btn")

        nsfw_button.click(fn=check_nsfw_image, inputs=image_input, outputs=nsfw_output)
        violence_button.click(fn=check_violence_image, inputs=image_input, outputs=violence_output)

    with gr.Tab("🌐 Kiểm duyệt URL"):
        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(label="Nhập đường dẫn URL cần kiểm tra")
            with gr.Column(scale=1):
                url_output = gr.Textbox(label="Kết quả kiểm duyệt URL")
                url_button = gr.Button("Kiểm tra URL", elem_classes="yellow-btn")
        url_button.click(fn=check_url, inputs=url_input, outputs=url_output)


