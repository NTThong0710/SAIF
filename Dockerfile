# Sử dụng image chính thức từ Hugging Face
FROM python:3.10-slim

# Cài đặt hệ thống và Python packages
RUN apt-get update && apt-get install -y git

# Tạo thư mục app
WORKDIR /app

# Copy tất cả file sang Docker container
COPY . .

# Cài requirements
RUN pip install --no-cache-dir -r requirements.txt

# Chạy app Gradio
CMD ["python", "saifguard_demo.py"]
