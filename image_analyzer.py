"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Sezgisel Görüntü Analizi

Model yalnızca yirmilik dişin kutusunu verdiği için buradaki sınıflamalar
görüntü işleme tabanlı *tahminlerdir*; hekim onayı gerekir. Bilinen sınırlar:
  - Winter açısı 2. molar ekseni yerine görüntü dikeyine göre ölçülür.
  - Kron ucu mine parlaklığından bulunur; belirsizse kronun oklüzal tarafta olduğu varsayılır.
  - Oklüzal düzlem, dişin mesialindeki dişlemler arası koyu boşluktan tahmin edilir.
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ========================================================================
# SABİTLER
# ========================================================================

JAW_UPPER = "Üst Çene"
JAW_LOWER = "Alt Çene"

RAMUS_NOT_APPLICABLE = "Uygulanamaz (Üst Çene)"

# Görüntü dikeyinden sapma (derece). Winter: dikey ±10°, yatay 80-100°;
# 2. molar ekseni yerine görüntü dikeyi kullanıldığı için biraz geniş tutuldu.
TILT_VERTICAL_MAX = 15
TILT_HORIZONTAL_MIN = 75
TILT_HORIZONTAL_MAX = 150  # bunun ötesi: kron alt çene kenarına / yukarı bakıyor → ters
# Kron ucunun (mine) kök ucundan en az bu oranda parlak olması gerekir; yoksa
# kronun oklüzal tarafta olduğu varsayılır.
ENAMEL_BRIGHTNESS_RATIO = 1.08

# Etiketli verideki üst/alt diş merkezlerinden kalibre edildi (y_c ≈ 0.52 → %91 doğruluk)
JAW_SPLIT_Y_RATIO = 0.52

# Segmentasyon parametreleri
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MIN_CONTOUR_AREA_RATIO = 0.15  # Kırpılmış ROI alanının minimum %15'i

# Ramus tespiti
RAMUS_SEARCH_WIDTH_RATIO = 1.2  # Diş genişliğinin katı — distal yönde arama
RAMUS_EDGE_THRESHOLD = 0.5
RAMUS_OVERLAP_THRESHOLD_CLASS2 = 0.25  # Dişin %25'i ramus içinde → Sınıf 2
RAMUS_OVERLAP_THRESHOLD_CLASS3 = 0.65  # Dişin %65'i ramus içinde → Sınıf 3

# Derinlik (diş yüksekliğine oranla, oklüzal düzlemden uzaklık)
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

    Returns:
        dict: {fdi, jaw, jaw_confidence, impaction, impaction_confidence, angle_value,
               depth, depth_confidence, ramus, ramus_confidence,
               tooth_axis, occlusal_y}
    """
    x1, y1, x2, y2 = map(int, bbox)
    img_h, img_w = image.shape[:2]
    bbox = [x1, y1, x2, y2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    others = [list(map(int, b)) for b in (all_bboxes or []) if not _same_box(b, bbox)]

    jaw, jaw_conf = detect_jaw(bbox, others, img_w, img_h)
    impaction, angle_val, angle_conf, axis = detect_tooth_angle(bbox, gray, img_w, jaw)
    depth, depth_conf, occlusal_y = detect_depth_level(bbox, gray, img_w, img_h, jaw, others)

    if jaw == JAW_UPPER:
        ramus, ramus_conf = RAMUS_NOT_APPLICABLE, jaw_conf
    else:
        ramus, ramus_conf = detect_ramus_relation(bbox, gray, img_w, img_h)

    return {
        "fdi": fdi_number(bbox, img_w, jaw),
        "jaw": jaw,
        "jaw_confidence": round(jaw_conf, 2),
        "impaction": impaction,
        "impaction_confidence": round(angle_conf, 2),
        "angle_value": round(angle_val, 1),
        "depth": depth,
        "depth_confidence": round(depth_conf, 2),
        "ramus": ramus,
        "ramus_confidence": round(ramus_conf, 2),
        "tooth_axis": axis,
        "occlusal_y": occlusal_y,
    }


def fdi_number(bbox, img_w, jaw):
    """
    FDI diş numarası. Panoramik röntgende görüntünün solu hastanın sağıdır:
    sol-üst 18, sağ-üst 28, sağ-alt 38, sol-alt 48.
    """
    if _is_left(bbox, img_w):
        return 18 if jaw == JAW_UPPER else 48
    return 28 if jaw == JAW_UPPER else 38


def _same_box(a, b):
    return all(abs(int(p) - int(q)) < 3 for p, q in zip(a, b))


def _is_left(bbox, img_w):
    return (bbox[0] + bbox[2]) / 2 < img_w / 2


# ========================================================================
# ÇENE (ÜST / ALT) TESPİTİ
# ========================================================================

def detect_jaw(bbox, others, img_w, img_h):
    """
    1. Aynı tarafta dikeyde ayrık başka diş varsa → göreli konum (en güvenilir)
    2. Karşı tarafta üst+alt çifti varsa → çiftin orta noktası eşik
    3. Aksi halde sabit oran eşiği (dişin mesialindeki koyu boşluğa bakmak
       ölçümde daha kötü çıktı: %80 vs %91)
    """
    x1, y1, x2, y2 = bbox
    cy = (y1 + y2) / 2
    h = y2 - y1
    left = _is_left(bbox, img_w)

    same_side = [b for b in others if _is_left(b, img_w) == left]
    for b in same_side:
        ocy = (b[1] + b[3]) / 2
        if abs(ocy - cy) > 0.4 * max(h, b[3] - b[1]):
            return (JAW_UPPER if cy < ocy else JAW_LOWER), 0.95

    other_side = sorted((b for b in others if _is_left(b, img_w) != left),
                        key=lambda b: (b[1] + b[3]) / 2)
    if len(other_side) >= 2:
        top, bottom = other_side[0], other_side[-1]
        t_cy, b_cy = (top[1] + top[3]) / 2, (bottom[1] + bottom[3]) / 2
        if b_cy - t_cy > 0.4 * max(top[3] - top[1], bottom[3] - bottom[1]):
            split = (t_cy + b_cy) / 2
            return (JAW_UPPER if cy < split else JAW_LOWER), 0.85

    return (JAW_UPPER if cy < img_h * JAW_SPLIT_Y_RATIO else JAW_LOWER), 0.6


# ========================================================================
# GÖMÜLÜLÜK AÇISI (WINTER)
# ========================================================================

def detect_tooth_angle(bbox, gray, img_w, jaw):
    """
    Diş kontürünün PCA ana ekseninden gömülülük açısını belirler.

    Returns:
        (sınıf_adı, dikeyden_sapma_derece, güvenilirlik, eksen_uç_noktaları)
        Sapma işareti: + → kron mesiale (ortaya) eğik, − → distale eğik.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return "Dikey (Vertical)", 0.0, 0.3, None

    pad_x, pad_y = int(bw * 0.10), int(bh * 0.08)
    cx1, cy1 = x1 + pad_x, y1 + pad_y
    roi = gray[cy1:y2 - pad_y, cx1:x2 - pad_x]
    if roi.size == 0 or min(roi.shape) < 10:
        return _angle_from_bbox_ratio(bw, bh)

    enhanced = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE).apply(roi)
    smoothed = cv2.bilateralFilter(enhanced, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    mask = _segment_tooth(smoothed)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = roi.shape[0] * roi.shape[1] * MIN_CONTOUR_AREA_RATIO
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return _angle_from_bbox_ratio(bw, bh)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, -1)
    ys, xs = np.nonzero(filled)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    elongation = math.sqrt(eigvals[1] / max(eigvals[0], 1e-6))

    # Oklüzal yön (görüntü koordinatında): alt çenede yukarı, üst çenede aşağı
    occlusal_dir = np.array([0.0, -1.0 if jaw == JAW_LOWER else 1.0])
    if major @ occlusal_dir < 0:
        major = -major  # varsayılan: kron oklüzal tarafta

    # Mine en radyoopak dokudur: eksenin öbür ucu belirgin şekilde daha parlaksa kron oradadır
    proj = (pts - mean) @ major
    vals = smoothed[ys, xs].astype(np.float64)
    span = proj.max() - proj.min()
    crown_end = vals[proj > proj.max() - span * 0.3].mean()
    root_end = vals[proj < proj.min() + span * 0.3].mean()
    enamel_flipped = root_end > crown_end * ENAMEL_BRIGHTNESS_RATIO
    if enamel_flipped:
        major = -major
        crown_end, root_end = root_end, crown_end

    # Kron yönünün oklüzal yönden sapması; + → mesiale (görüntü ortasına) doğru
    mesial_sign = 1.0 if _is_left(bbox, img_w) else -1.0
    cos_a = float(np.clip(major @ occlusal_dir, -1.0, 1.0))
    side = mesial_sign * major[0]
    mesial_tilt = math.degrees(math.acos(cos_a)) * (1.0 if side >= 0 else -1.0)

    classification = _classify_tilt(mesial_tilt)

    # Güvenilirlik: dişin ne kadar uzun/ince bulunduğu, sınıf sınırına uzaklık
    # ve kron ucunun mine parlaklığıyla ne kadar net ayrıldığı
    abs_tilt = abs(mesial_tilt)
    margin = min(abs(abs_tilt - t) for t in (TILT_VERTICAL_MAX, TILT_HORIZONTAL_MIN, TILT_HORIZONTAL_MAX))
    shape_score = min(1.0, max(0.0, (elongation - 1.1) / 1.0))
    enamel_score = min(1.0, max(0.0, (crown_end / max(root_end, 1.0) - 1.0) / 0.15))
    confidence = 0.25 + 0.3 * shape_score + 0.2 * min(1.0, margin / 15) + 0.15 * enamel_score
    if enamel_flipped and abs_tilt > TILT_HORIZONTAL_MIN:
        confidence = min(confidence, 0.5)  # kron yönü ters çevrildi — hekim kontrolü şart

    # axis[0] = kök ucu, axis[1] = kron ucu
    half = 0.5 * math.sqrt(eigvals[1]) * 2.5
    cx, cy = mean[0] + cx1, mean[1] + cy1
    axis = [[int(cx - major[0] * half), int(cy - major[1] * half)],
            [int(cx + major[0] * half), int(cy + major[1] * half)]]

    return classification, mesial_tilt, min(confidence, 0.9), axis


def _classify_tilt(mesial_tilt):
    abs_tilt = abs(mesial_tilt)
    if abs_tilt <= TILT_VERTICAL_MAX:
        return "Dikey (Vertical)"
    if abs_tilt > TILT_HORIZONTAL_MAX:
        return "Ters (Inverted)"
    if abs_tilt >= TILT_HORIZONTAL_MIN:
        return "Yatay (Horizontal)"
    return "Mesioangular" if mesial_tilt > 0 else "Distoangular"


def _segment_tooth(roi_gray):
    """Diş (parlak) bölgesini Otsu eşiği ile ayırır."""
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def _angle_from_bbox_ratio(bw, bh):
    """Segmentasyon başarısız olursa kutu oranından kaba tahmin (yön bilinmez)."""
    ratio = bw / bh if bh > 0 else 1.0
    if ratio > 1.5:
        return "Yatay (Horizontal)", 90.0, 0.35, None
    return "Dikey (Vertical)", 0.0, 0.3, None


# ========================================================================
# RAMUS İLİŞKİSİ (yalnızca alt çene)
# ========================================================================

def detect_ramus_relation(bbox, gray, img_w, img_h):
    """
    Dişin distalindeki bölgede en güçlü dikey kenarı (ramus ön kenarı) arar.
    Arama alanı dişin kendi distal kenarının biraz içinden başlar, böylece
    dişin kendi kenarı ramus sanılmaz.

    Returns:
        (sınıf_adı, güvenilirlik)
    """
    x1, y1, x2, y2 = bbox
    tooth_w, tooth_h = x2 - x1, y2 - y1
    distal_is_left = _is_left(bbox, img_w)
    search_w = max(int(tooth_w * RAMUS_SEARCH_WIDTH_RATIO), 30)
    inset = int(tooth_w * 0.6)  # dişin distal %40'ı + distalindeki alan

    if distal_is_left:
        sx1, sx2 = max(0, x1 - search_w), x1 + tooth_w - inset
    else:
        sx1, sx2 = x2 - tooth_w + inset, min(img_w, x2 + search_w)
    sy1 = max(0, y1 - int(tooth_h * 0.3))
    sy2 = min(img_h, y2 + int(tooth_h * 0.3))

    if sx2 - sx1 < 10 or sy2 - sy1 < 10:
        return _ramus_fallback(bbox, img_w)

    edge_x = _find_ramus_edge(gray[sy1:sy2, sx1:sx2], distal_is_left)
    if edge_x is None:
        return _ramus_fallback(bbox, img_w)

    ramus_abs_x = sx1 + edge_x
    if distal_is_left:
        overlap = (ramus_abs_x - x1) / tooth_w
    else:
        overlap = (x2 - ramus_abs_x) / tooth_w
    overlap = min(1.0, max(0.0, overlap))

    if overlap >= RAMUS_OVERLAP_THRESHOLD_CLASS3:
        return "Sınıf 3 (Tam Ramus İçinde)", 0.5
    if overlap >= RAMUS_OVERLAP_THRESHOLD_CLASS2:
        return "Sınıf 2 (Yarı Ramus İçinde)", 0.45
    return "Sınıf 1 (Önünde)", 0.5


def _find_ramus_edge(roi, distal_is_left):
    """
    ROI'de ramus ön kenarını bulur: ramus (parlak kemik) distal tarafta olduğundan
    distal→mesial yönünde parlaktan koyuya geçen en güçlü dikey kenar aranır.
    """
    if roi.size == 0:
        return None
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(roi)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    # distal solda ise parlak→koyu geçiş soldan sağa negatif gradyan demektir
    profile = (-sobel_x if distal_is_left else sobel_x).mean(axis=0)
    profile = np.clip(profile, 0, None)
    if profile.max() <= 0:
        return None
    profile = profile / profile.max()
    strong = np.where(profile > RAMUS_EDGE_THRESHOLD)[0]
    if len(strong) == 0:
        return None
    # mesiale en yakın güçlü kenar = ramus ön kenarı
    return int(strong[-1] if distal_is_left else strong[0])


def _ramus_fallback(bbox, img_w):
    """Kenar bulunamadığında konum tabanlı kaba tahmin."""
    relative_x = ((bbox[0] + bbox[2]) / 2) / img_w
    if relative_x < 0.15 or relative_x > 0.85:
        return "Sınıf 2 (Yarı Ramus İçinde)", 0.3
    return "Sınıf 1 (Önünde)", 0.3


# ========================================================================
# GÖMÜLÜLÜK DERİNLİĞİ (PELL & GREGORY A/B/C)
# ========================================================================

def detect_depth_level(bbox, gray, img_w, img_h, jaw, others=None):
    """
    Dişin oklüzal yüzeyinin, 2. molar bölgesindeki oklüzal düzleme göre konumu.

    Returns:
        (sınıf_adı, güvenilirlik, oklüzal_y)
    """
    x1, y1, x2, y2 = bbox
    tooth_h = max(1, y2 - y1)

    occlusal_y, ref_conf = _estimate_occlusal_plane(bbox, gray, img_w, img_h, jaw, others or [])

    # Pozitif = diş oklüzal düzlemden uzakta (daha derin)
    if jaw == JAW_LOWER:
        relative_depth = (y1 - occlusal_y) / tooth_h
    else:
        relative_depth = (occlusal_y - y2) / tooth_h

    if relative_depth < DEPTH_LEVEL_A_RATIO:
        classification = "Seviye A (Oklüzal)"
    elif relative_depth < DEPTH_LEVEL_B_RATIO:
        classification = "Seviye B (Oklüzal-Servikal Arası)"
    else:
        classification = "Seviye C (Servikal Altı - Derin)"

    logger.debug("Derinlik: jaw=%s occlusal_y=%d relative=%.2f → %s",
                 jaw, occlusal_y, relative_depth, classification)
    return classification, ref_conf, int(occlusal_y)


def _estimate_occlusal_plane(bbox, gray, img_w, img_h, jaw, others):
    """
    Dişin mesialindeki şeritte (2. molar bölgesi) satır ortalama parlaklığının
    en koyu olduğu satırı — alt ve üst dişler arasındaki boşluğu — bulur.
    Aynı tarafta karşı çenede diş varsa arama iki dişin merkezleri arasıyla sınırlanır.
    """
    x1, y1, x2, y2 = bbox
    tooth_w, tooth_h = x2 - x1, y2 - y1

    if _is_left(bbox, img_w):
        sx1, sx2 = x2, min(img_w, x2 + int(tooth_w * 1.2))
    else:
        sx1, sx2 = max(0, x1 - int(tooth_w * 1.2)), x1

    # Oklüzal düzlem alt çenede dişin üstünde, üst çenede altında aranır
    if jaw == JAW_LOWER:
        sy1, sy2 = y1 - int(tooth_h * 0.8), y1 + int(tooth_h * 0.5)
    else:
        sy1, sy2 = y2 - int(tooth_h * 0.5), y2 + int(tooth_h * 0.8)

    cy = (y1 + y2) / 2
    left = _is_left(bbox, img_w)
    for b in others:
        ocy = (b[1] + b[3]) / 2
        if _is_left(b, img_w) != left or abs(ocy - cy) < 0.4 * tooth_h:
            continue
        if jaw == JAW_LOWER and ocy < cy:
            sy1 = max(sy1, int(ocy))
        elif jaw == JAW_UPPER and ocy > cy:
            sy2 = min(sy2, int(ocy))
    sy1, sy2 = max(0, sy1), min(img_h, sy2)

    fallback_y = y1 - 0.1 * tooth_h if jaw == JAW_LOWER else y2 + 0.1 * tooth_h
    if sx2 - sx1 < 10 or sy2 - sy1 < 10:
        return fallback_y, 0.3

    strip = cv2.GaussianBlur(gray[sy1:sy2, sx1:sx2], (5, 5), 0).astype(np.float64)
    profile = strip.mean(axis=1)
    k = max(3, (sy2 - sy1) // 25)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    # Kenarlardaki konvolüsyon etkisini dışla
    core = profile[k:-k] if len(profile) > 2 * k + 5 else profile
    offset = k if len(profile) > 2 * k + 5 else 0
    idx = int(np.argmin(core)) + offset

    contrast = (np.median(profile) - profile[idx]) / (profile.std() + 1e-6)
    confidence = float(min(0.8, max(0.3, 0.3 + 0.2 * contrast)))
    return float(sy1 + idx), confidence
