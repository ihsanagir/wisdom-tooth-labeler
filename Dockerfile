FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (OpenCV ve PyTorch için gerekli)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch'un CPU sürümü (CUDA'lı sürüm imajı birkaç GB büyütür, sunucuda GPU yok)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Import doğrulama (Eksik C-kütüphanesi varsa build aşamasında yakalar)
RUN python -c "import app; print('✅ App import testi basarili!')"

# Uygulama başlatma
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
