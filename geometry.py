"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Anatomik nokta geometrisi

Hekimin (veya ileride pose modelinin) işaretlediği anatomik noktalardan
Winter açısı, Pell & Gregory derinliği/ramus sınıfı ve sinir yakınlığını hesaplar.
Görüntü koordinatları piksel cinsindendir (y aşağı doğru artar).

Noktalar (her yirmilik diş için):
    m3_crown        M3 kron tepesi (oklüzale en yakın nokta)
    m3_apex         M3 kök apeksi (çok köklüde köklerin orta noktası)
    m2_cusp_mesial  2. molar mesial tüberkül tepesi
    m2_cusp_distal  2. molar distal tüberkül tepesi
    m2_cej_distal   2. molar distal mine-sement sınırı (servikal hat)
    m2_apex         2. molar kök apeksi
    ramus_anterior  Ramus ön kenarı, oklüzal düzlem hizasında (yalnız alt çene)
    canal_top       Mandibular kanal üst sınırı, M3 apeksi hizasında (yalnız alt çene)
"""

import math

KEYPOINTS = [
    "m3_crown", "m3_apex",
    "m2_cusp_mesial", "m2_cusp_distal", "m2_cej_distal", "m2_apex",
    "ramus_anterior", "canal_top",
]

KEYPOINT_LABELS = {
    "m3_crown": "M3 kron tepesi",
    "m3_apex": "M3 kök apeksi",
    "m2_cusp_mesial": "M2 mesial tüberkül",
    "m2_cusp_distal": "M2 distal tüberkül",
    "m2_cej_distal": "M2 distal servikal hat (CEJ)",
    "m2_apex": "M2 kök apeksi",
    "ramus_anterior": "Ramus ön kenarı",
    "canal_top": "Mandibular kanal üst sınırı",
}

LOWER_ONLY = {"ramus_anterior", "canal_top"}

# Winter sınıfları (M2 eksenine göre sapma, derece)
WINTER_VERTICAL_MAX = 10
WINTER_HORIZONTAL_MIN = 80
WINTER_INVERTED_MIN = 150

# P&G ramus: (M2 distali ile ramus arası boşluk) / (M3 kron genişliği)
RAMUS_CLASS1_MIN_RATIO = 1.0
RAMUS_CLASS3_MAX_RATIO = 0.15

# Sinir: apeks–kanal mesafesi M3 boyunun bu oranından azsa "yakın" (~2 mm)
NERVE_CLOSE_RATIO = 0.10


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _norm(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def applicable_keypoints(jaw):
    """Üst çenede ramus ve mandibular kanal noktaları yoktur."""
    return [k for k in KEYPOINTS if jaw != "Üst Çene" or k not in LOWER_ONLY]


def _occlusal_direction(kp, jaw):
    """M2'nin apeksten oklüzale uzanan birim ekseni; apeks yoksa çeneye göre dikey."""
    if all(kp.get(k) for k in ("m2_apex", "m2_cusp_mesial", "m2_cusp_distal")):
        occl_mid = _mid(kp["m2_cusp_mesial"], kp["m2_cusp_distal"])
        return _norm(_vec(kp["m2_apex"], occl_mid)), True
    return ((0.0, -1.0) if jaw != "Üst Çene" else (0.0, 1.0)), False


def winter_angle(kp, jaw):
    """
    M3 ekseninin (apeks→kron) M2 eksenine göre sapması.
    + değer: kron mesiale (M2'ye) doğru eğik.
    """
    if not (kp.get("m3_crown") and kp.get("m3_apex")):
        return None
    m2_axis, from_m2 = _occlusal_direction(kp, jaw)
    m3_axis = _norm(_vec(kp["m3_apex"], kp["m3_crown"]))

    angle = math.degrees(math.acos(max(-1.0, min(1.0, _dot(m2_axis, m3_axis)))))

    # Mesial yön: M3'ten M2'ye doğru (yatay bileşen)
    ref = None
    for k in ("m2_cusp_mesial", "m2_cusp_distal", "m2_apex"):
        if kp.get(k):
            ref = kp[k]
            break
    if ref is not None:
        mesial = (1.0 if ref[0] > kp["m3_apex"][0] else -1.0, 0.0)
        if _dot(m3_axis, mesial) < _dot(m2_axis, mesial):
            angle = -angle

    abs_a = abs(angle)
    if abs_a <= WINTER_VERTICAL_MAX:
        cls = "Dikey (Vertical)"
    elif abs_a >= WINTER_INVERTED_MIN:
        cls = "Ters (Inverted)"
    elif abs_a >= WINTER_HORIZONTAL_MIN:
        cls = "Yatay (Horizontal)"
    else:
        cls = "Mesioangular" if angle > 0 else "Distoangular"

    return {"value": cls, "angle": round(angle, 1), "reference": "M2 ekseni" if from_m2 else "görüntü dikeyi"}


def pell_gregory_depth(kp, jaw):
    """M3'ün en oklüzal noktasının M2 oklüzal düzlemi ve servikal hattına göre seviyesi."""
    need = ("m3_crown", "m2_cusp_mesial", "m2_cusp_distal", "m2_cej_distal")
    if not all(kp.get(k) for k in need):
        return None
    u, _ = _occlusal_direction(kp, jaw)
    occl_mid = _mid(kp["m2_cusp_mesial"], kp["m2_cusp_distal"])
    d_crown = _dot(_vec(occl_mid, kp["m3_crown"]), u)   # ≥0: oklüzal düzlemde/üstünde
    d_cej = _dot(_vec(occl_mid, kp["m2_cej_distal"]), u)  # negatif

    if d_crown >= 0:
        cls = "Seviye A (Oklüzal)"
    elif d_crown >= d_cej:
        cls = "Seviye B (Oklüzal-Servikal Arası)"
    else:
        cls = "Seviye C (Servikal Altı - Derin)"
    return {"value": cls, "crown_offset_px": round(d_crown, 1), "cej_offset_px": round(d_cej, 1)}


def pell_gregory_ramus(kp, jaw, m3_width):
    """M2 distal yüzü ile ramus ön kenarı arasındaki boşluğun M3 kron genişliğine oranı."""
    if jaw == "Üst Çene":
        return {"value": "Uygulanamaz (Üst Çene)"}
    if not (kp.get("m2_cusp_distal") and kp.get("ramus_anterior") and m3_width):
        return None
    # Distal yön: M2'den ramusa doğru
    distal_sign = 1.0 if kp["ramus_anterior"][0] > kp["m2_cusp_distal"][0] else -1.0
    if kp.get("m2_cusp_mesial"):
        distal_sign = 1.0 if kp["m2_cusp_distal"][0] > kp["m2_cusp_mesial"][0] else -1.0

    space = (kp["ramus_anterior"][0] - kp["m2_cusp_distal"][0]) * distal_sign
    ratio = space / m3_width
    if ratio >= RAMUS_CLASS1_MIN_RATIO:
        cls = "Sınıf 1 (Önünde)"
    elif ratio > RAMUS_CLASS3_MAX_RATIO:
        cls = "Sınıf 2 (Yarı Ramus İçinde)"
    else:
        cls = "Sınıf 3 (Tam Ramus İçinde)"
    return {"value": cls, "space_ratio": round(ratio, 2)}


def nerve_proximity(kp, jaw):
    """M3 apeksi ile mandibular kanal üst sınırı arasındaki mesafe (M3 boyuna oranla)."""
    if jaw == "Üst Çene":
        return None
    if not all(kp.get(k) for k in ("m3_crown", "m3_apex", "canal_top")):
        return None
    length = math.dist(kp["m3_crown"], kp["m3_apex"])
    if length == 0:
        return None
    gap = kp["canal_top"][1] - kp["m3_apex"][1]  # pozitif: kanal apeksin altında
    ratio = gap / length
    cls = "Yakın/Temaslı" if ratio < NERVE_CLOSE_RATIO else "Uzak"
    return {"value": cls, "gap_ratio": round(ratio, 2)}


def analyze_keypoints(keypoints, jaw, bbox=None):
    """
    Args:
        keypoints: {isim: [x, y]} — eksik noktalar atlanabilir
        jaw: "Alt Çene" | "Üst Çene"
        bbox: [x1, y1, x2, y2] — M3 kron genişliği için
    Returns:
        {impaction, depth, ramus, nerve}: her biri {value, ...} veya None,
        ve "missing": hesap için eksik noktalar
    """
    kp = {k: tuple(v) for k, v in (keypoints or {}).items() if v and k in KEYPOINTS}
    width = (bbox[2] - bbox[0]) if bbox else None
    result = {
        "impaction": winter_angle(kp, jaw),
        "depth": pell_gregory_depth(kp, jaw),
        "ramus": pell_gregory_ramus(kp, jaw, width),
        "nerve": nerve_proximity(kp, jaw),
    }
    result["missing"] = [k for k in applicable_keypoints(jaw) if k not in kp]
    return result
