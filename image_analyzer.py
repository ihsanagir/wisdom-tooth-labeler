
import cv2
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# ========================================================================
# SABİTLER
# ========================================================================

# Açı sınıflandırma eşikleri (derece — dikey eksene göre)
ANGLE_VERTICAL_MIN = 70
ANGLE_VERTICAL_MAX = 110
ANGLE_HORIZONTAL_LOW = 25
ANGLE_HORIZONTAL_HIGH = 155
ANGLE_INVERTED_MIN = 120

# Segmentasyon parametreleri
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = 5
MORPH_KERNEL_SIZE = 3
MIN_CONTOUR_AREA_RATIO = 0.05  # Bbox alanının minimum %5'i

# Ramus tespiti
RAMUS_SEARCH_WIDTH_RATIO = 0.15  # Dişin arkasında araştırma genişliği
SOBEL_KSIZE = 3
RAMUS_EDGE_THRESHOLD = 0.4  # Sobel sonucu normalize eşiği
RAMUS_OVERLAP_THRESHOLD_CLASS2 = 0.25  # Dişin %25'i ramus içinde → Sınıf 2
RAMUS_OVERLAP_THRESHOLD_CLASS3 = 0.65  # Dişin %65'i ramus içinde → Sınıf 3

# Derinlik tespiti
DEPTH_LEVEL_A_RATIO = 0.15  # Oklüzal düzleme çok yakın
DEPTH_LEVEL_B_RATIO = 0.40  # Oklüzal-servikal arası


# ========================================================================
# ANA FONKSİYON
# ========================================================================

def analyze_tooth_automatically(bbox, image, all_bboxes=None):
    """
    Tek bir diş için tüm otomatik analizleri yapar.

    Args:
        bbox: [x1, y1, x2, y2] — dişin bounding box'ı
        image: numpy array — tam görüntü
        all_bboxes: list of [x1,y1,x2,y2] — tüm tespit edilen dişlerin bbox'ları
                    (derinlik ve ramus analizi için referans)

    Returns:
        dict: {impaction, impaction_confidence, angle_value,
               depth, depth_confidence, ramus, ramus_confidence}
    """
    x1, y1, x2, y2 = map(int, bbox)
    img_h, img_w = image.shape[:2]

    # Gri tonlama
    if len(image.shape) == 3:
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = image.copy()

    # --- Açı Tespiti ---
    impaction, angle_val, angle_conf = detect_tooth_angle(
        [x1, y1, x2, y2], gray_full, img_w, img_h
    )

    # --- Derinlik Tespiti ---
    depth, depth_conf = detect_depth_level(
        [x1, y1, x2, y2], gray_full, img_h, all_bboxes
    )

    # --- Ramus İlişkisi ---
    ramus, ramus_conf = detect_ramus_relation(
        [x1, y1, x2, y2], gray_full, img_w, img_h
    )

    return {
        "impaction": impaction,
        "impaction_confidence": round(angle_conf, 2),
        "angle_value": round(angle_val, 1),
        "depth": depth,
        "depth_confidence": round(depth_conf, 2),
        "ramus": ramus,
        "ramus_confidence": round(ramus_conf, 2),
    }


# ========================================================================
# GÖMÜLÜLGİÇ AÇISI TESPİTİ
# ========================================================================

def detect_tooth_angle(bbox, gray, img_w, img_h):
    """
    Diş kontürünün minAreaRect açısını kullanarak gömülülük açısını belirler.

    Yöntem:
    1. Bbox'u hafif küçülterek merkezdeki dişe odaklan
    2. CLAHE + Bilateral Filter ile ön işleme
    3. Adaptive Threshold ile segmentasyon
    4. Morfolojik temizleme → en büyük kontür
    5. minAreaRect → açı
    6. Bbox aspect ratio ile çapraz doğrulama

    Returns:
        (sınıf_adı, açı_değeri, güvenilirlik)
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return "Dikey (Vertical)", 90, 0.3

    # --- İç bölge kesimi (kenarları %10 kırp — kemik/komşu dişi azalt) ---
    pad_x = int(bw * 0.10)
    pad_y = int(bh * 0.08)
    cx1 = max(0, x1 + pad_x)
    cy1 = max(0, y1 + pad_y)
    cx2 = min(img_w, x2 - pad_x)
    cy2 = min(img_h, y2 - pad_y)

    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    roi = gray[cy1:cy2, cx1:cx2]

    if roi.size == 0:
        return "Dikey (Vertical)", 90, 0.3

    # --- Ön İşleme ---
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced = clahe.apply(roi)
    smoothed = cv2.bilateralFilter(enhanced, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    # --- Segmentasyon ---
    tooth_mask = _segment_tooth(smoothed)

    # --- Kontür bulma ---
    contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: Bbox oranına dayalı tahmin
        return _angle_from_bbox_ratio(bw, bh, x1, x2, img_w)

    # En büyük kontürü seç
    min_area = (cx2 - cx1) * (cy2 - cy1) * MIN_CONTOUR_AREA_RATIO
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return _angle_from_bbox_ratio(bw, bh, x1, x2, img_w)

    largest_contour = max(valid_contours, key=cv2.contourArea)

    # --- minAreaRect ile açı ---
    if len(largest_contour) < 5:
        return _angle_from_bbox_ratio(bw, bh, x1, x2, img_w)

    rect = cv2.minAreaRect(largest_contour)
    rect_w, rect_h = rect[1]
    rect_angle = rect[2]  # -90 ile 0 arası

    # minAreaRect açısını normalize et
    # OpenCV4: uzun kenar her zaman width, açı -90..0
    # Amacımız: dişin dikey eksene göre açısını bulmak (0°=yatay, 90°=dikey)
    if rect_w < rect_h:
        # Uzun kenar dikey
        tooth_angle = abs(rect_angle)  # 0'a yakınsa → dikey
    else:
        # Uzun kenar yatay
        tooth_angle = 90 + rect_angle  # Saat yönünde çevirme

    # 0-180 aralığına normalize et
    tooth_angle = tooth_angle % 180

    # Dikey eksene göre açı: 90° = tam dikey
    # Dişin sol/sağ tarafını belirle
    tooth_center_x = (x1 + x2) / 2
    is_left_side = tooth_center_x < (img_w / 2)

    # --- Bbox oranı ile çapraz doğrulama ---
    bbox_angle_est, _, bbox_conf = _angle_from_bbox_ratio(bw, bh, x1, x2, img_w)

    # Kontür ve bbox tahmini arasındaki tutarlılık → güvenilirlik
    consistency = _angle_consistency(tooth_angle, bbox_angle_est, bw, bh)

    # Açıyı ağırlıklı birleştir (kontür ağırlığı daha yüksek)
    if consistency > 0.7:
        confidence = min(0.95, 0.6 + consistency * 0.35)
    else:
        confidence = max(0.35, consistency * 0.6)
        # Düşük tutarlılıkta bbox'a daha çok güven
        if bbox_conf > 0.5:
            tooth_angle = tooth_angle * 0.6 + _bbox_angle_numeric(bw, bh) * 0.4

    # --- Sınıflandırma ---
    classification = _classify_angle(tooth_angle, is_left_side)

    return classification, tooth_angle, confidence


def _segment_tooth(roi_gray):
    """
    Gri tonlamalı ROI'dan diş bölgesini segmente eder.

    Returns:
        binary mask (uint8)
    """
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )

    # Morfolojik temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ters çevir (diş = beyaz, arka plan = siyah) — röntgende diş genelde parlak
    # Ama adaptive threshold ile diş beyaz gelmiş olabilir
    # İstatistiksel kontrol: merkezin değerine bak
    h, w = cleaned.shape
    center_val = cleaned[h // 2, w // 2]
    if center_val == 0:
        cleaned = cv2.bitwise_not(cleaned)

    return cleaned


def _angle_from_bbox_ratio(bw, bh, x1, x2, img_w):
    """
    Bbox width/height oranından fallback açı tahmini yapar.

    Returns:
        (sınıf_adı, açı_değeri, güvenilirlik)
    """
    ratio = bw / bh if bh > 0 else 1.0

    if ratio > 1.5:
        return "Yatay (Horizontal)", 10, 0.55
    elif ratio > 1.15:
        # Mesio/Disto — tarafa bağlı
        center_x = (x1 + x2) / 2
        is_left = center_x < (img_w / 2)
        angle = 45
        if is_left:
            return "Mesioangular", angle, 0.40
        else:
            return "Mesioangular", angle, 0.40
    elif ratio < 0.65:
        return "Dikey (Vertical)", 88, 0.50
    else:
        return "Dikey (Vertical)", 85, 0.45


def _bbox_angle_numeric(bw, bh):
    """Bbox oranından sayısal açı tahmini (0-180)."""
    ratio = bw / bh if bh > 0 else 1.0
    if ratio > 1.5:
        return 15  # Yatay
    elif ratio > 1.1:
        return 45  # Eğik
    elif ratio < 0.7:
        return 90  # Dikey
    else:
        return 80  # Dikeye yakın


def _angle_consistency(contour_angle, bbox_class, bw, bh):
    """
    Kontür açısı ile bbox sınıflandırması arasındaki tutarlılığı ölçer.
    Returns: 0.0-1.0 (1.0 = tam tutarlı)
    """
    bbox_angle = _bbox_angle_numeric(bw, bh)

    diff = abs(contour_angle - bbox_angle)
    if diff > 90:
        diff = 180 - diff

    # 0-30° fark: yüksek tutarlılık, 30-60°: orta, 60+: düşük
    if diff < 20:
        return 1.0
    elif diff < 40:
        return 0.7
    elif diff < 60:
        return 0.4
    else:
        return 0.2


def _classify_angle(angle_deg, is_left_side):
    """
    Açı değerinden sınıflandırma yapar.

    Args:
        angle_deg: 0-180 arası açı (0=yatay, 90=dikey)
        is_left_side: Dişin görüntünün sol yarısında olup olmadığı

    Returns:
        str: Sınıf adı
    """
    # Dikey
    if ANGLE_VERTICAL_MIN <= angle_deg <= ANGLE_VERTICAL_MAX:
        return "Dikey (Vertical)"

    # Yatay
    if angle_deg < ANGLE_HORIZONTAL_LOW or angle_deg > ANGLE_HORIZONTAL_HIGH:
        return "Yatay (Horizontal)"

    # Ters
    if angle_deg > ANGLE_INVERTED_MIN:
        return "Ters (Inverted)"

    # Mesioangular vs Distoangular
    # Panoramik röntgende:
    #   Sol yarı = Hastanın sağ çenesi
    #   Sağ yarı = Hastanın sol çenesi
    #
    # Mesioangular: Dişin tepesi komşu dişe (ortaya) doğru eğik
    # Distoangular: Dişin tepesi arkaya (ramusa) doğru eğik

    if is_left_side:
        # Sol yarıdaki diş → Sağ çene
        # Açı < 70°: eğim sağa (ortaya) → Mesioangular
        # Açı 110-120°: eğim sola (arkaya) → Distoangular
        if angle_deg < ANGLE_VERTICAL_MIN:
            return "Mesioangular"
        else:
            return "Distoangular"
    else:
        # Sağ yarıdaki diş → Sol çene
        # Açı > 110°: eğim sola (ortaya) → Mesioangular
        # Açı < 70°: eğim sağa (arkaya) → Distoangular
        if angle_deg > ANGLE_VERTICAL_MAX:
            return "Mesioangular"
        else:
            return "Distoangular"


# ========================================================================
# RAMUS İLİŞKİSİ TESPİTİ
# ========================================================================

def detect_ramus_relation(bbox, gray, img_w, img_h):
    """
    Dişin ramus (çene dalı) ile ilişkisini belirler.

    Yöntem:
    1. Dişin posterior (arka) tarafını belirle
    2. Arkadaki bölgede Sobel-Y ile dikey kenarları tespit et
    3. En güçlü dikey kenar = ramus ön kenarı
    4. Dişin ramus kenarına göre pozisyonunu sınıflandır

    Returns:
        (sınıf_adı, güvenilirlik)
    """
    x1, y1, x2, y2 = bbox
    tooth_w = x2 - x1
    tooth_center_x = (x1 + x2) / 2
    is_left_side = tooth_center_x < (img_w / 2)

    # --- Ramus arama bölgesini belirle ---
    search_w = max(int(img_w * RAMUS_SEARCH_WIDTH_RATIO), 30)

    if is_left_side:
        # Sol taraftaki diş: ramus daha solda (x1'in solunda)
        search_x1 = max(0, x1 - search_w)
        search_x2 = x1 + int(tooth_w * 0.3)  # Dişin biraz içine de bak
        ramus_direction = "left"  # Ramus solda
    else:
        # Sağ taraftaki diş: ramus daha sağda (x2'nin sağında)
        search_x1 = x2 - int(tooth_w * 0.3)
        search_x2 = min(img_w, x2 + search_w)
        ramus_direction = "right"  # Ramus sağda

    # Dikey arama alanını genişlet (diş yüksekliğinin %50 üstü ve altı)
    tooth_h = y2 - y1
    search_y1 = max(0, y1 - int(tooth_h * 0.3))
    search_y2 = min(img_h, y2 + int(tooth_h * 0.3))

    if search_x2 - search_x1 < 10 or search_y2 - search_y1 < 10:
        return "Sınıf 1 (Önünde)", 0.30

    search_roi = gray[search_y1:search_y2, search_x1:search_x2]

    # --- Sobel ile dikey kenar tespiti ---
    ramus_edge_x = _find_ramus_edge(search_roi, ramus_direction)

    if ramus_edge_x is None:
        # Ramus kenarı bulunamadı — konum tabanlı fallback
        return _ramus_fallback(bbox, img_w)

    # Ramus kenarının gerçek x koordinatı
    ramus_abs_x = search_x1 + ramus_edge_x

    # --- Sınıflandırma ---
    # Dişin ramus kenarına göre ne kadar içeride olduğunu hesapla
    if is_left_side:
        # Sol taraf: ramus solda, diş sağda
        # Dişin sol kenarı (x1) vs ramus kenarı
        if x1 >= ramus_abs_x:
            # Diş tamamen ramus önünde
            overlap = 0.0
        else:
            # Dişin bir kısmı ramus içinde
            overlap_px = ramus_abs_x - x1
            overlap = overlap_px / tooth_w
    else:
        # Sağ taraf: ramus sağda, diş solda
        # Dişin sağ kenarı (x2) vs ramus kenarı
        if x2 <= ramus_abs_x:
            # Diş tamamen ramus önünde
            overlap = 0.0
        else:
            # Dişin bir kısmı ramus içinde
            overlap_px = x2 - ramus_abs_x
            overlap = overlap_px / tooth_w

    overlap = min(1.0, max(0.0, overlap))

    # Sınıflandır
    if overlap >= RAMUS_OVERLAP_THRESHOLD_CLASS3:
        classification = "Sınıf 3 (Tam Ramus İçinde)"
        confidence = 0.55 + min(0.35, (overlap - RAMUS_OVERLAP_THRESHOLD_CLASS3) * 2)
    elif overlap >= RAMUS_OVERLAP_THRESHOLD_CLASS2:
        classification = "Sınıf 2 (Yarı Ramus İçinde)"
        confidence = 0.50 + min(0.35, (overlap - RAMUS_OVERLAP_THRESHOLD_CLASS2) * 1.5)
    else:
        classification = "Sınıf 1 (Önünde)"
        confidence = 0.60 + min(0.30, (1 - overlap) * 0.4)

    logger.debug(
        "Ramus: side=%s, ramus_x=%d, overlap=%.2f → %s (conf=%.2f)",
        "left" if is_left_side else "right", ramus_abs_x, overlap,
        classification, confidence
    )

    return classification, confidence


def _find_ramus_edge(roi, direction):
    """
    ROI içinde ramus ön kenarını (en güçlü dikey kenar) bulur.

    Args:
        roi: grayscale numpy array
        direction: "left" veya "right" — ramus hangi tarafta

    Returns:
        int: ramus kenarının x koordinatı (ROI içinde), veya None
    """
    if roi.size == 0:
        return None

    # CLAHE ile kontrast artır
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(roi)

    # Blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Sobel-X ile dikey kenarları bul (mutlak değer)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobel_abs = np.abs(sobel_x)

    # Her sütunun toplam kenar gücünü hesapla (dikey kenar profili)
    edge_profile = sobel_abs.mean(axis=0)

    if edge_profile.max() == 0:
        return None

    # Normalize et
    edge_profile = edge_profile / edge_profile.max()

    # Eşik üstü güçlü kenarları bul
    strong_edges = np.where(edge_profile > RAMUS_EDGE_THRESHOLD)[0]

    if len(strong_edges) == 0:
        return None

    # Ramus kenarı: sağ tarafta ise en soldaki güçlü kenar,
    # sol tarafta ise en sağdaki güçlü kenar
    if direction == "left":
        # Ramus solda → en sağdaki güçlü kenar = ramus ön kenarı
        ramus_x = strong_edges[-1]
    else:
        # Ramus sağda → en soldaki güçlü kenar = ramus ön kenarı
        ramus_x = strong_edges[0]

    return int(ramus_x)


def _ramus_fallback(bbox, img_w):
    """
    Ramus kenarı bulunamadığında konum tabanlı fallback.
    Mevcut yöntemin biraz geliştirilmiş hali.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    relative_x = center_x / img_w

    if relative_x < 0.15 or relative_x > 0.85:
        return "Sınıf 2 (Yarı Ramus İçinde)", 0.35
    elif relative_x < 0.25 or relative_x > 0.75:
        return "Sınıf 1 (Önünde)", 0.40
    else:
        return "Sınıf 1 (Önünde)", 0.30


# ========================================================================
# GÖMÜLÜLGİÇ DERİNLİĞİ TESPİTİ
# ========================================================================

def detect_depth_level(bbox, gray, img_h, all_bboxes=None):
    """
    Dişin gömülülük derinliğini belirler (Pell & Gregory A/B/C).

    Yöntem:
    1. Diğer tespit edilen dişlerden oklüzal düzlem referansı oluştur
    2. Yoksa: Lokal yoğunluk profilinden oklüzal düzlem tahmini
    3. Dişin üst kenarını oklüzal düzlemle karşılaştır

    Returns:
        (sınıf_adı, güvenilirlik)
    """
    x1, y1, x2, y2 = bbox
    tooth_h = y2 - y1
    tooth_top = y1

    # --- Oklüzal düzlem referansı ---
    occlusal_y, ref_confidence = _estimate_occlusal_plane(
        bbox, img_h, all_bboxes
    )

    # --- Dişin oklüzal yüzeyinin referansa göre pozisyonu ---
    # Pozitif = diş oklüzal düzlemin altında (daha derin)
    relative_depth = (tooth_top - occlusal_y) / tooth_h if tooth_h > 0 else 0

    # Sınıflandırma
    if relative_depth < DEPTH_LEVEL_A_RATIO:
        # Dişin üstü oklüzal düzlem seviyesinde veya üstünde
        classification = "Seviye A (Oklüzal)"
        depth_conf = ref_confidence * 0.9
    elif relative_depth < DEPTH_LEVEL_B_RATIO:
        # Oklüzal düzlem ile servikal çizgi arası
        classification = "Seviye B (Oklüzal-Servikal Arası)"
        depth_conf = ref_confidence * 0.85
    else:
        # Servikal çizginin altında
        classification = "Seviye C (Servikal Altı - Derin)"
        depth_conf = ref_confidence * 0.80

    logger.debug(
        "Derinlik: occlusal_y=%d, tooth_top=%d, relative=%.2f → %s (conf=%.2f)",
        occlusal_y, tooth_top, relative_depth, classification, depth_conf
    )

    return classification, depth_conf


def _estimate_occlusal_plane(target_bbox, img_h, all_bboxes=None):
    """
    Oklüzal düzlem y-koordinatını tahmin eder.

    Öncelik sırası:
    1. Diğer dişlerin y1 medyanı (en güvenilir)
    2. Tek diş varsa: bbox oranına dayalı tahmin

    Returns:
        (y_koordinatı, güvenilirlik)
    """
    tx1, ty1, tx2, ty2 = target_bbox

    if all_bboxes and len(all_bboxes) > 1:
        # --- Yöntem 1: Diğer dişlerin üst kenarlarının medyanı ---
        other_tops = []
        target_center_y = (ty1 + ty2) / 2

        for b in all_bboxes:
            bx1, by1, bx2, by2 = map(int, b)
            b_center_y = (by1 + by2) / 2

            # Aynı diş mi? (IoU kontrolü)
            if abs(bx1 - tx1) < 20 and abs(by1 - ty1) < 20:
                continue

            # Aynı çene yarısında mı? (üst/alt çene yakınlığı)
            # y ekseni farkı çok büyükse farklı çenede olabilir
            if abs(b_center_y - target_center_y) < img_h * 0.3:
                other_tops.append(by1)

        if other_tops:
            occlusal_y = int(np.median(other_tops))
            confidence = min(0.90, 0.50 + len(other_tops) * 0.15)
            return occlusal_y, confidence

    # --- Yöntem 2: Tek diş — bbox ve görüntü oranına dayalı ---
    # Diş Y pozisyonuna göre: üst bölgede ise muhtemelen yüzeysel
    relative_y = ty1 / img_h

    if relative_y < 0.35:
        # Görüntünün üst kısmında — muhtemelen oklüzal düzlem yakın
        occlusal_y = int(ty1 - (ty2 - ty1) * 0.1)
    elif relative_y < 0.55:
        # Orta bölge
        occlusal_y = int(ty1 - (ty2 - ty1) * 0.2)
    else:
        # Alt bölge — muhtemelen daha derin
        occlusal_y = int(ty1 - (ty2 - ty1) * 0.35)

    return max(0, occlusal_y), 0.40  # Tek dişte güvenilirlik düşük
