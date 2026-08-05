FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (OpenCV için gerekli)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Uygulama başlatma
# Uygulama başlatma — Railway PORT env variable'ını okur
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
