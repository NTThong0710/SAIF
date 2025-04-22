import gradio as gr
from app.safety_check import is_prompt_safe
from app.gen_ai import generate_response
from app.mlops_logger import log_prompt

def handle_prompt(prompt):
    safe, info = is_prompt_safe(prompt)
    if not safe:
        log_prompt(prompt, info, False, "")
        return f"🚨 Prompt không an toàn! Phát hiện: {', '.join(info)}", ""
    
    response = generate_response(prompt)
    log_prompt(prompt, "OK", True, response)
    return "✅ Prompt an toàn", response

demo = gr.Interface(
    fn=handle_prompt,
    inputs=gr.Textbox(label="Nhập Prompt"),
    outputs=[
        gr.Textbox(label="Trạng thái kiểm duyệt"),
        gr.Textbox(label="Kết quả GenAI")
    ],
    title="SAIFGuard: GenAI Prompt Checker",
    description="Kiểm tra prompt an toàn và tạo văn bản bằng GPT2"
)
