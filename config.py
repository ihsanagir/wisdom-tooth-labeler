"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Konfigürasyon
"""
import os

APP_VERSION = "1.2.0"

# --- Model Ayarları ---
# Model seçimi (2026-10-08): dataset_v2 test setinin hiçbir modelin eğitimde görmediği
# 35 görüntülük alt kümesinde disprojesi52 en yüksek mAP50 (0.988) ve recall (0.983).
# Sızıntısız bölmeyle eğitilen disprojesi6: mAP50 0.953 — daha iyi değil.
# disprojesi2/3 train==valid bölmesiyle eğitildi → adil ölçülemez, kullanılmamalı.
MODEL_PATH = os.getenv("MODEL_PATH", "trained_models/disprojesi52/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.35  # 0.30 → 0.35 (post-filter ile birlikte daha dengeli)
# Sunucuda GPU yok; lokalde de eğitimle GPU belleği paylaşılmasın diye varsayılan CPU
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")
MAX_DETECTIONS = 4  # Maksimum tespit edilecek diş sayısı (post-filter de ayrıca sınırlar)

# --- Sunucu Ayarları ---
HOST = "0.0.0.0"          # Railway için tüm IP'lere açık
PORT = int(os.getenv("PORT", 7860))  # Railway PORT env değişkenini otomatik atar

# --- Dropdown Seçenekleri ---
GENDER_OPTIONS = ["Erkek", "Kadın"]
AGE_OPTIONS = ["<20", "20-25", "26-30", ">30"]
MOUTH_OPENING_OPTIONS = ["Normal (>40mm)", "Kısıtlı (30-40mm)", "Çok Kısıtlı (<30mm)"]
IMPACTION_OPTIONS = ["Dikey (Vertical)", "Mesioangular", "Distoangular", "Yatay (Horizontal)", "Ters (Inverted)"]
RAMUS_OPTIONS = ["Sınıf 1 (Önünde)", "Sınıf 2 (Yarı Ramus İçinde)", "Sınıf 3 (Tam Ramus İçinde)", "Uygulanamaz (Üst Çene)"]
DEPTH_OPTIONS = ["Seviye A (Oklüzal)", "Seviye B (Oklüzal-Servikal Arası)", "Seviye C (Servikal Altı - Derin)"]
ROOT_OPTIONS = ["Normal/Konik", "Eğri/Dilasere", "Ayrık/Diverjan"]
NERVE_OPTIONS = ["Uzak", "Yakın/Temaslı"]
