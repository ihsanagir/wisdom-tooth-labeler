"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Zorluk Değerlendirmesi

Temel ölçüt: Pederson zorluk indeksi (Pederson, 1988) — Winter açısı,
Pell & Gregory derinliği ve ramus ilişkisinden 3-10 arası puan:
    3-4  : az zor      5-6 : orta zor      7-10 : çok zor

Hasta/anatomik risk faktörleri için literatürde doğrulanmış puan ağırlıkları
olmadığından puana eklenmez; bunun yerine açık bir kuralla kategoriyi yükseltir:
    1 büyük risk  veya  2+ orta risk  → kategori bir seviye yükselir.
"""

from config import (
    GENDER_OPTIONS, AGE_OPTIONS, MOUTH_OPENING_OPTIONS,
    IMPACTION_OPTIONS, RAMUS_OPTIONS, DEPTH_OPTIONS,
    ROOT_OPTIONS, NERVE_OPTIONS,
)

# Seçenek listeleriyle aynı sırada: Dikey, Mesioangular, Distoangular, Yatay, Ters
# Ters (inverted) Pederson'un orijinal tablosunda yok; en zor sınıf olarak 4 puan.
IMPACTION_POINTS = dict(zip(IMPACTION_OPTIONS, [3, 1, 4, 2, 4]))
# Sınıf 1/2/3; üst çenede ramus ilişkisi yok → minimum puan ve uyarı
RAMUS_POINTS = dict(zip(RAMUS_OPTIONS, [1, 2, 3, 1]))
DEPTH_POINTS = dict(zip(DEPTH_OPTIONS, [1, 2, 3]))

SEVERITY_LEVELS = ["simple", "surgical", "advanced"]
SEVERITY_LABELS = {
    "simple": "Az Zor",
    "surgical": "Orta Zor",
    "advanced": "Çok Zor",
}
RECOMMENDATIONS = {
    "simple": (
        "✅ BASİT ÇEKİM / MİNİMAL CERRAHİ\n"
        "Genellikle flep kaldırmadan veya küçük bir flep ile çekilebilir."
    ),
    "surgical": (
        "⚖️ CERRAHİ ÇEKİM (OPERASYON)\n"
        "Flep kaldırılması, kemik alınması (osteotomi) ve dişin "
        "bölünmesi (odontotomi) gerekebilir."
    ),
    "advanced": (
        "🚨 İLERİ CERRAHİ / UZMAN GÖRÜŞÜ\n"
        "Yüksek komplikasyon riski. Ağız, Diş ve Çene Cerrahisi uzmanı tarafından "
        "değerlendirilmesi önerilir."
    ),
}

_ALLOWED = {
    "gender": GENDER_OPTIONS,
    "age": AGE_OPTIONS,
    "mouth_opening": MOUTH_OPENING_OPTIONS,
    "impaction": IMPACTION_OPTIONS,
    "ramus": RAMUS_OPTIONS,
    "depth": DEPTH_OPTIONS,
    "root": ROOT_OPTIONS,
    "nerve": NERVE_OPTIONS,
}


def validate_inputs(**values):
    """Bilinmeyen seçenek değerlerini döndürür (boş liste = geçerli)."""
    return [
        f"{field}: '{value}'"
        for field, value in values.items()
        if value not in _ALLOWED[field]
    ]


def _risk_factors(age, mouth_opening, root, nerve):
    """(faktör, değer, seviye, açıklama) listesi. seviye: 'major' | 'moderate'"""
    risks = []
    if nerve == "Yakın/Temaslı":
        risks.append(("Sinir (IAN) komşuluğu", nerve, "major",
                      "⚠️ Sinir (IAN) komşuluğu/teması riski! CBCT ile teyit önerilir."))
    if mouth_opening == "Çok Kısıtlı (<30mm)":
        risks.append(("Ağız açıklığı", mouth_opening, "major",
                      "Ciddi açıklık kısıtlılığı! Genel anestezi gerekebilir."))
    elif mouth_opening == "Kısıtlı (30-40mm)":
        risks.append(("Ağız açıklığı", mouth_opening, "moderate",
                      "Kısıtlı ağız açıklığı çalışmayı zorlaştırabilir."))
    if root in ("Eğri/Dilasere", "Ayrık/Diverjan"):
        risks.append(("Kök formu", root, "moderate",
                      "Kök morfolojisi kök kırığı ve kemik kaldırma ihtiyacını artırabilir."))
    if age == ">30":
        risks.append(("Yaş", age, "moderate",
                      "30 yaş üstünde kemik yoğunluğu ve iyileşme süresi zorluğu artırabilir."))
    return risks


def analyze_case(gender, age, mouth_opening, impaction, ramus, depth, root, nerve):
    """
    Pederson indeksi + risk faktörlerine göre zorluk kategorisi ve öneri üretir.
    Girdilerin validate_inputs ile doğrulanmış olması beklenir.
    """
    breakdown = [
        {"factor": "Gömülülük açısı", "value": impaction, "points": IMPACTION_POINTS[impaction]},
        {"factor": "Derinlik", "value": depth, "points": DEPTH_POINTS[depth]},
        {"factor": "Ramus ilişkisi", "value": ramus, "points": RAMUS_POINTS[ramus]},
    ]
    index = sum(item["points"] for item in breakdown)

    if index <= 4:
        base_severity = "simple"
    elif index <= 6:
        base_severity = "surgical"
    else:
        base_severity = "advanced"

    risks = _risk_factors(age, mouth_opening, root, nerve)
    n_major = sum(1 for r in risks if r[2] == "major")
    n_moderate = sum(1 for r in risks if r[2] == "moderate")
    escalated = n_major >= 1 or n_moderate >= 2

    level = SEVERITY_LEVELS.index(base_severity)
    severity = SEVERITY_LEVELS[min(level + 1, 2)] if escalated else base_severity

    warnings = [r[3] for r in risks]
    notes = []
    if ramus == "Uygulanamaz (Üst Çene)":
        notes.append("Pederson indeksi alt çene için geliştirilmiştir; üst çenede ramus "
                     "puanı minimum (1) alındı, sonuç yaklaşık değerdir.")
    if impaction == "Ters (Inverted)":
        notes.append("Ters pozisyon Pederson tablosunda yer almaz; en zor sınıf (4) olarak puanlandı.")
    if escalated and severity != base_severity:
        notes.append(f"Risk faktörleri nedeniyle kategori '{SEVERITY_LABELS[base_severity]}' → "
                     f"'{SEVERITY_LABELS[severity]}' yükseltildi.")

    recommendation = RECOMMENDATIONS[severity]
    if warnings:
        recommendation += "\n\n📋 Klinik Notlar:\n" + "\n".join(warnings)

    return {
        "pederson_index": index,
        "pederson_max": 10,
        "breakdown": breakdown,
        "base_severity": base_severity,
        "severity": severity,
        "severity_label": SEVERITY_LABELS[severity],
        "escalated": escalated and severity != base_severity,
        "risk_factors": [
            {"factor": r[0], "value": r[1], "level": r[2]} for r in risks
        ],
        "recommendation": recommendation,
        "warnings": warnings,
        "notes": notes,
    }
