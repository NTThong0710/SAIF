with gr.Blocks(title="SAIFGuard") as demo:
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

# Thêm CSS để tạo nút màu vàng
css = """
.yellow-btn {
    background-color: #FFD700 !important;
    color: black !important;
}
"""

demo = gr.Blocks(css=css)