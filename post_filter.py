"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Post-Detection Anatomik Filtreleme
Yanlış pozitif (Tip 2) hataları azaltmak için anatomik bilgi tabanlı filtreleme.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Anatomik Filtreleme Sabitleri ---
# Yirmilik dişler panoramik filmde kenar bölgelerinde yer alır
WISDOM_ZONE_LEFT_MAX = 0.32   # Görüntünün sol %32'si
WISDOM_ZONE_RIGHT_MIN = 0.68  # Görüntünün sağ %68'inden sonrası
MIDDLE_ZONE_MIN_CONF = 0.75   # Orta bölgedeki tespitler için minimum güven

# Kadran başına maksimum diş sayısı (üst + alt çene)
MAX_TEETH_PER_HALF = 2

# Minimum bbox alan oranı (görüntü alanına göre)
MIN_BBOX_AREA_RATIO = 0.002

# Çok yakın tespitler arası minimum mesafe (normalize, görüntü genişliğine göre)
MIN_DISTANCE_RATIO = 0.05


def filter_wisdom_detections(detections, image_width, image_height):
    """
    Anatomik bilgi tabanlı yanlış pozitif filtreleme.

    Kurallar:
    1. Bölge filtresi: Orta bölgedeki tespitler yüksek güven gerektirir
    2. Kadran sınırlaması: Her yarıda maksimum 2 yirmilik diş
    3. Boyut filtresi: Çok küçük tespitleri at
    4. Overlap filtresi: Çok yakın tespitlerde düşük güvenli olanı at

    Args:
        detections: list of dict, her biri {"bbox": [x1,y1,x2,y2], "confidence": float, ...}
        image_width: int
        image_height: int

    Returns:
        filtered: list of dict - filtrelenmiş tespitler
        removed: list of dict - kaldırılan tespitler (loglama için)
    """
    if not detections:
        return [], []

    filtered = []
    removed = []

    # --- 1. Boyut Filtresi ---
    image_area = image_width * image_height
    size_filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / image_area

        if area_ratio < MIN_BBOX_AREA_RATIO:
            det["filter_reason"] = f"Çok küçük bbox (alan oranı: {area_ratio:.4f})"
            removed.append(det)
            logger.info("Filtre [Boyut]: Diş %d kaldırıldı - %s", det.get("index", "?"), det["filter_reason"])
        else:
            size_filtered.append(det)

    # --- 2. Bölge Filtresi ---
    zone_filtered = []
    for det in size_filtered:
        x1, y1, x2, y2 = det["bbox"]
        center_x = (x1 + x2) / 2
        relative_x = center_x / image_width

        in_wisdom_zone = (relative_x < WISDOM_ZONE_LEFT_MAX) or (relative_x > WISDOM_ZONE_RIGHT_MIN)

        if not in_wisdom_zone:
            # Orta bölgede: yüksek güven gerektirir
            if det["confidence"] < MIDDLE_ZONE_MIN_CONF:
                det["filter_reason"] = (
                    f"Orta bölge + düşük güven (x: {relative_x:.2f}, conf: {det['confidence']:.2f})"
                )
                removed.append(det)
                logger.info("Filtre [Bölge]: Diş %d kaldırıldı - %s", det.get("index", "?"), det["filter_reason"])
                continue

        zone_filtered.append(det)

    # --- 3. Overlap/Yakınlık Filtresi ---
    proximity_filtered = _remove_close_duplicates(zone_filtered, image_width)
    for det in zone_filtered:
        if det not in proximity_filtered:
            det["filter_reason"] = "Yakın tespitte düşük güvenli olan"
            removed.append(det)
            logger.info("Filtre [Yakınlık]: Diş %d kaldırıldı", det.get("index", "?"))

    # --- 4. Kadran Sınırlaması ---
    left_teeth = [d for d in proximity_filtered if (d["bbox"][0] + d["bbox"][2]) / 2 < image_width / 2]
    right_teeth = [d for d in proximity_filtered if (d["bbox"][0] + d["bbox"][2]) / 2 >= image_width / 2]

    # Her yarıda en yüksek güvenli MAX_TEETH_PER_HALF dişi tut
    left_teeth.sort(key=lambda d: d["confidence"], reverse=True)
    right_teeth.sort(key=lambda d: d["confidence"], reverse=True)

    kept_left = left_teeth[:MAX_TEETH_PER_HALF]
    kept_right = right_teeth[:MAX_TEETH_PER_HALF]

    for det in left_teeth[MAX_TEETH_PER_HALF:]:
        det["filter_reason"] = f"Sol yarıda kadran sınırı aşıldı (maks {MAX_TEETH_PER_HALF})"
        removed.append(det)
        logger.info("Filtre [Kadran]: Diş %d kaldırıldı - %s", det.get("index", "?"), det["filter_reason"])

    for det in right_teeth[MAX_TEETH_PER_HALF:]:
        det["filter_reason"] = f"Sağ yarıda kadran sınırı aşıldı (maks {MAX_TEETH_PER_HALF})"
        removed.append(det)
        logger.info("Filtre [Kadran]: Diş %d kaldırıldı - %s", det.get("index", "?"), det["filter_reason"])

    filtered = kept_left + kept_right

    # İndeksleri yeniden numaralandır
    filtered.sort(key=lambda d: d["bbox"][0])  # Soldan sağa sırala
    for i, det in enumerate(filtered):
        det["index"] = i + 1

    logger.info(
        "Filtreleme tamamlandı: %d → %d tespit (%d kaldırıldı)",
        len(detections), len(filtered), len(removed)
    )

    return filtered, removed


def _remove_close_duplicates(detections, image_width):
    """
    Birbirine çok yakın tespitlerde düşük güvenli olanı kaldırır.
    """
    if len(detections) <= 1:
        return list(detections)

    keep = list(detections)
    to_remove = set()

    for i in range(len(keep)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(keep)):
            if j in to_remove:
                continue

            ci_x = (keep[i]["bbox"][0] + keep[i]["bbox"][2]) / 2
            cj_x = (keep[j]["bbox"][0] + keep[j]["bbox"][2]) / 2
            ci_y = (keep[i]["bbox"][1] + keep[i]["bbox"][3]) / 2
            cj_y = (keep[j]["bbox"][1] + keep[j]["bbox"][3]) / 2

            dist = ((ci_x - cj_x) ** 2 + (ci_y - cj_y) ** 2) ** 0.5
            dist_ratio = dist / image_width

            if dist_ratio < MIN_DISTANCE_RATIO:
                # Daha düşük güvenli olanı kaldır
                if keep[i]["confidence"] < keep[j]["confidence"]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    return [det for idx, det in enumerate(keep) if idx not in to_remove]
