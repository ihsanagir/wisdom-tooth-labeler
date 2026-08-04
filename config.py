"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Konfigürasyon
"""
import os

# --- Model Ayarları ---
# Yeni eğitim sonrası: "trained_models/disprojesi4/weights/best.pt" olarak güncellenecek
MODEL_PATH = "trained_models/disprojesi3/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.35  # 0.30 → 0.35 (post-filter ile birlikte daha dengeli)
MAX_DETECTIONS = 4  # Maksimum tespit edilecek diş sayısı (post-filter de ayrıca sınırlar)

# --- Sunucu Ayarları ---
HOST = "0.0.0.0"          # Railway için tüm IP'lere açık
PORT = int(os.getenv("PORT", 7860))  # Railway PORT env değişkenini otomatik atar

# --- Skor Sabitleri ---
BASE_SCORE = 20  # Taban zorluk puanı

SCORE_WEIGHTS = {
    # Hasta Faktörleri
    "gender_male": 2,
    "age_26_30": 10,
    "age_over_30": 20,
    "mouth_limited": 15,
    "mouth_very_limited": 30,

    # Diş Pozisyonu (Gömülülük Açısı)
    "impaction_mesioangular": 15,
    "impaction_distoangular": 30,
    "impaction_horizontal": 40,
    "impaction_inverted": 50,

    # Ramus İlişkisi
    "ramus_class2": 15,
    "ramus_class3": 30,

    # Derinlik
    "depth_level_b": 15,
    "depth_level_c": 30,

    # Kök ve Anatomik Riskler
    "root_curved": 20,
    "root_divergent": 25,
    "nerve_close": 30,
}

# --- Dropdown Seçenekleri ---
GENDER_OPTIONS = ["Erkek", "Kadın"]
AGE_OPTIONS = ["<20", "20-25", "26-30", ">30"]
MOUTH_OPENING_OPTIONS = ["Normal (>40mm)", "Kısıtlı (30-40mm)", "Çok Kısıtlı (<30mm)"]
IMPACTION_OPTIONS = ["Dikey (Vertical)", "Mesioangular", "Distoangular", "Yatay (Horizontal)", "Ters (Inverted)"]
RAMUS_OPTIONS = ["Sınıf 1 (Önünde)", "Sınıf 2 (Yarı Ramus İçinde)", "Sınıf 3 (Tam Ramus İçinde)"]
DEPTH_OPTIONS = ["Seviye A (Oklüzal)", "Seviye B (Oklüzal-Servikal Arası)", "Seviye C (Servikal Altı - Derin)"]
ROOT_OPTIONS = ["Normal/Konik", "Eğri/Dilasere", "Ayrık/Diverjan"]
NERVE_OPTIONS = ["Uzak", "Yakın/Temaslı"]
