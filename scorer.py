"""
Akıllı Yirmilik Diş Karar Destek Sistemi — Zorluk Skoru Hesaplama
Pell & Gregory sınıflandırması tabanlı klinik karar destek modülü.
"""

from config import SCORE_WEIGHTS, BASE_SCORE


def analyze_case(gender, age, mouth_opening, impaction, ramus, depth, root, nerve):
    """
    Kullanıcı girdilerine göre zorluk derecesini hesaplar ve tedavi önerisi sunar.

    Returns:
        dict: {
            "score": int (0-100),
            "recommendation": str,
            "severity": str ("simple" | "surgical" | "advanced"),
            "warnings": list[str]
        }
    """
    if not all([gender, age, mouth_opening, impaction, ramus, depth, root, nerve]):
        return {
            "score": 0,
            "recommendation": "Lütfen tüm alanları eksiksiz doldurunuz.",
            "severity": "unknown",
            "warnings": []
        }

    score = 0
    warnings = []
    w = SCORE_WEIGHTS

    # --- 1. Hasta Faktörleri ---
    if gender == "Erkek":
        score += w["gender_male"]
    if age == "26-30":
        score += w["age_26_30"]
    elif age == ">30":
        score += w["age_over_30"]

    if mouth_opening == "Kısıtlı (30-40mm)":
        score += w["mouth_limited"]
        warnings.append("Kısıtlı ağız açıklığı çalışmayı zorlaştırabilir.")
    elif mouth_opening == "Çok Kısıtlı (<30mm)":
        score += w["mouth_very_limited"]
        warnings.append("Ciddi açıklık kısıtlılığı! Genel anestezi gerekebilir.")

    # --- 2. Diş Pozisyonu (Pell & Gregory) ---
    impaction_map = {
        "Mesioangular": w["impaction_mesioangular"],
        "Distoangular": w["impaction_distoangular"],
        "Yatay (Horizontal)": w["impaction_horizontal"],
        "Ters (Inverted)": w["impaction_inverted"],
    }
    score += impaction_map.get(impaction, 0)

    ramus_map = {
        "Sınıf 2 (Yarı Ramus İçinde)": w["ramus_class2"],
        "Sınıf 3 (Tam Ramus İçinde)": w["ramus_class3"],
    }
    score += ramus_map.get(ramus, 0)

    depth_map = {
        "Seviye B (Oklüzal-Servikal Arası)": w["depth_level_b"],
        "Seviye C (Servikal Altı - Derin)": w["depth_level_c"],
    }
    score += depth_map.get(depth, 0)

    # --- 3. Kök ve Anatomik Riskler ---
    root_map = {
        "Eğri/Dilasere": w["root_curved"],
        "Ayrık/Diverjan": w["root_divergent"],
    }
    score += root_map.get(root, 0)

    if nerve == "Yakın/Temaslı":
        score += w["nerve_close"]
        warnings.append("⚠️ DİKKAT: Sinir (IAN) komşuluğu/teması riski! CBCT ile teyit önerilir.")

    # --- Sonuç ---
    final_score = min(100, max(0, BASE_SCORE + score))

    if final_score < 40:
        severity = "simple"
        recommendation = (
            "✅ BASİT ÇEKİM / MİNİMAL CERRAHİ\n"
            "Genellikle flep kaldırmadan veya küçük bir flep ile çekilebilir."
        )
    elif final_score < 70:
        severity = "surgical"
        recommendation = (
            "⚖️ CERRAHİ ÇEKİM (OPERASYON)\n"
            "Flep kaldırılması, kemik alınması (osteoektomi) ve dişin "
            "bölünmesi (odontektomi) gerekebilir."
        )
    else:
        severity = "advanced"
        recommendation = (
            "🚨 İLERİ CERRAHİ / UZMAN GÖRÜŞÜ\n"
            "Yüksek komplikasyon riski. Çene Cerrahisi Uzmanı tarafından "
            "değerlendirilmesi önerilir."
        )

    if warnings:
        recommendation += "\n\n📋 Klinik Notlar:\n" + "\n".join(warnings)

    return {
        "score": final_score,
        "recommendation": recommendation,
        "severity": severity,
        "warnings": warnings,
    }
