from detoxify import Detoxify
from transformers import (
    AutoProcessor, AutoModelForImageClassification,
    ViTForImageClassification, ViTFeatureExtractor,
    BlipProcessor, BlipForConditionalGeneration, pipeline
)
from PIL import Image
import torch

import re
from urllib.parse import urlparse, unquote

# Load model phát hiện URL độc hại
classifier = pipeline("zero-shot-classification")

# Load models
detox_model = Detoxify('original')

nsfw_model_id = "Falconsai/nsfw_image_detection"
nsfw_processor = AutoProcessor.from_pretrained(nsfw_model_id)
nsfw_model = AutoModelForImageClassification.from_pretrained(nsfw_model_id)

violence_model_id = "jaranohaal/vit-base-violence-detection"
violence_model = ViTForImageClassification.from_pretrained(violence_model_id)
violence_processor = ViTFeatureExtractor.from_pretrained(violence_model_id)

# Load BLIP cho caption của ảnh
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    if flagged:
        return False, list(flagged.keys())
    return True, []

def generate_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def check_nsfw_image(image: Image.Image) -> str:
    """Kiểm tra và trả về kết quả NSFW của ảnh"""
    # Xử lý NSFW
    nsfw_inputs = nsfw_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        nsfw_outputs = nsfw_model(**nsfw_inputs)
        nsfw_probs = torch.nn.functional.softmax(nsfw_outputs.logits, dim=1)[0]
    
    nsfw_labels = list(nsfw_model.config.id2label.values())
    nsfw_pred = nsfw_probs.argmax().item()
    nsfw_label = nsfw_labels[nsfw_pred]
    nsfw_score = nsfw_probs[nsfw_pred].item() * 100
    
    # Tạo caption
    caption = generate_caption(image)
    
    # Đánh giá kết quả
    if nsfw_label.lower() in ["porn", "hentai", "sex", "nsfw"]:
        return f"""🚨 Ảnh KHÔNG an toàn (NSFW):
- Loại: {nsfw_label}
- Độ chính xác: {nsfw_score:.2f}%
- Mô tả: {caption}"""
    else:
        return f"""✅ Ảnh an toàn (NSFW):
- Loại: {nsfw_label}
- Độ chính xác: {nsfw_score:.2f}%
- Mô tả: {caption}"""

def check_violence_image(image: Image.Image) -> str:
    """Kiểm tra và trả về kết quả bạo lực của ảnh"""
    # Xử lý bạo lực
    violence_inputs = violence_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        violence_outputs = violence_model(**violence_inputs)
        violence_probs = torch.nn.functional.softmax(violence_outputs.logits, dim=1)[0]
    
    violence_labels = ["Non-Violent", "Violent"]
    violence_pred = violence_probs.argmax().item()
    violence_label = violence_labels[violence_pred]
    violence_score = violence_probs[violence_pred].item() * 100
    
    # Tạo caption
    caption = generate_caption(image)
    
    # Đánh giá kết quả
    is_violent = False
    if violence_label.lower() == "non-violent" and violence_score > 50:
        is_violent = True
    elif violence_label.lower() == "violent" and violence_score > 80:
        is_violent = True
    
    if is_violent:
        return f"""🚨 Ảnh KHÔNG an toàn (Bạo lực):
- Loại: {violence_label}
- Độ chính xác: {violence_score:.2f}%
- Mô tả: {caption}"""
    else:
        return f"""✅ Ảnh an toàn (Bạo lực):
- Loại: {violence_label}
- Độ chính xác: {violence_score:.2f}%
- Mô tả: {caption}"""

# ===Hàm check url===
def check_url(url: str):
    try:
        # Chuẩn hóa URL (decode các ký tự đặc biệt)
        decoded_url = unquote(url)
        parsed = urlparse(decoded_url)
        
        # Danh sách cảnh báo
        warnings = []
        
        # 1. Phát hiện IP thay vì domain (http://203.0.113.45/...)
        if re.match(r'^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', decoded_url):
            warnings.append("🚨 Nguy hiểm: Truy cập trực tiếp bằng IP (thường dùng cho tấn công)")
        
        # 2. Phát hiện file thực thi (gift-card.exe)
        if re.search(r'\.(exe|msi|bat|js|jar|apk|dmg)(\?|$)', parsed.path.lower()):
            warnings.append("🚨 Nguy hiểm: URL chứa file thực thi có thể độc hại")
        
        # 3. Phát hiện redirect độc hại (redirect?target=...)
        if 'redirect' in parsed.path.lower() or 'url=' in parsed.query.lower():
            warnings.append("⚠️ Cảnh báo: URL chứa chức năng redirect (có thể lừa đảo)")
        
        # 4. Phát hiện ký tự đặc biệt (/login%20%2F%00%3F%2F%2E%2E)
        if re.search(r'%[0-9a-f]{2}|[\x00-\x1f\x7f]', url):
            warnings.append("🚨 Nguy hiểm: URL chứa ký tự mã hóa đáng ngờ (có thể tấn công)")
        
        # 5. Phát hiện domain giả mạo (example.com@malicious-site.com)
        if '@' in parsed.netloc:
            warnings.append("🚨 Lừa đảo: URL chứa kỹ thuật giả mạo domain (user@fake-domain)")
        
        # 6. Phát hiện domain giả danh (secure.example-login.com)
        deceptive_domains = ['login', 'secure', 'account', 'verify', 'update']
        if any(keyword in parsed.netloc.lower() for keyword in deceptive_domains):
            warnings.append("⚠️ Cảnh báo: Domain có dấu hiệu giả mạo dịch vụ đăng nhập")
        
        # 7. Kiểm tra giao thức không mã hóa
        if parsed.scheme == 'http':
            warnings.append("⚠️ Cảnh báo: Kết nối không mã hóa (HTTP)")
        
        # Kết hợp với AI classifier
        ai_result = classifier(url, candidate_labels=["malicious", "safe"])
        ai_label = ai_result["labels"][0]
        ai_score = ai_result["scores"][0] * 100
        
        # Tạo báo cáo
        report = {
            "url": url,
            "decoded_url": decoded_url,
            "domain": parsed.netloc,
            "path": parsed.path,
            "warnings": warnings,
            "ai_analysis": {
                "label": ai_label,
                "confidence": ai_score
            }
        }
        
        # Quyết định cuối cùng
        if warnings or ai_label == "malicious":
            return format_report(report, is_safe=False)
        else:
            return format_report(report, is_safe=True)
            
    except Exception as e:
        return f"⚠️ Lỗi khi phân tích URL: {str(e)}"

def format_report(report: dict, is_safe: bool):
    """Định dạng báo cáo dễ đọc"""
    warning_text = "\n".join(f"- {w}" for w in report["warnings"]) if report["warnings"] else "- Không phát hiện cảnh báo"
    
    if not is_safe:
        return f"""🚨 URL KHÔNG AN TOÀN
🔍 Phân tích chi tiết:
• URL gốc: {report['url']}
• Domain: {report['domain']}
• Đường dẫn: {report['path']}

📢 CẢNH BÁO:
{warning_text}

🤖 Phân tích AI:
- Kết quả: {report['ai_analysis']['label']}
- Độ tin cậy: {report['ai_analysis']['confidence']:.2f}%

🛡️ Khuyến nghị: KHÔNG TRUY CẬP!"""
    else:
        return f"""✅ URL AN TOÀN
🔍 Phân tích chi tiết:
• URL gốc: {report['url']}
• Domain: {report['domain']}

🤖 Phân tích AI:
- Kết quả: {report['ai_analysis']['label']}
- Độ tin cậy: {report['ai_analysis']['confidence']:.2f}%"""