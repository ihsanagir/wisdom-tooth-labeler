import math

import pytest

from geometry import analyze_keypoints, applicable_keypoints, winter_angle

LOWER = "Alt Çene"
UPPER = "Üst Çene"

# Görüntünün solunda alt çene (hasta sağı, 48): mesial = +x, M2 M3'ün sağında
M2_LOWER_LEFT = {
    "m2_cusp_mesial": (330, 400),
    "m2_cusp_distal": (290, 400),
    "m2_cej_distal": (285, 430),
    "m2_apex": (310, 520),
}

# Görüntünün sağında üst çene (hasta solu, 28): mesial = -x, kron aşağı bakar
M2_UPPER_RIGHT = {
    "m2_cusp_mesial": (880, 320),
    "m2_cusp_distal": (920, 320),
    "m2_cej_distal": (925, 290),
    "m2_apex": (900, 200),
}


def tilted(apex, length, deg, toward_x, crown_dir_y):
    """apex'ten başlayıp kronun oklüzal yöne göre `deg` derece `toward_x` yönüne eğik olduğu nokta."""
    r = math.radians(deg)
    return (apex[0] + toward_x * length * math.sin(r), apex[1] + crown_dir_y * length * math.cos(r))


@pytest.mark.parametrize("deg,toward,expected", [
    (0, 1, "Dikey (Vertical)"),
    (45, 1, "Mesioangular"),     # kron mesiale (+x)
    (45, -1, "Distoangular"),    # kron distale (-x)
    (90, 1, "Yatay (Horizontal)"),
    (180, 1, "Ters (Inverted)"),
])
def test_winter_lower_left(deg, toward, expected):
    apex = (220, 520)
    kp = {**M2_LOWER_LEFT, "m3_apex": apex, "m3_crown": tilted(apex, 120, deg, toward, -1)}
    w = winter_angle(kp, LOWER)
    assert w["value"] == expected
    assert abs(abs(w["angle"]) - deg) < 0.5
    assert w["reference"] == "M2 ekseni"


@pytest.mark.parametrize("deg,toward,expected", [
    (0, -1, "Dikey (Vertical)"),
    (45, -1, "Mesioangular"),    # üst sağda mesial = -x
    (45, 1, "Distoangular"),
])
def test_winter_upper_right(deg, toward, expected):
    apex = (980, 200)
    kp = {**M2_UPPER_RIGHT, "m3_apex": apex, "m3_crown": tilted(apex, 120, deg, toward, 1)}
    assert winter_angle(kp, UPPER)["value"] == expected


@pytest.mark.parametrize("crown_y,expected", [
    (395, "Seviye A (Oklüzal)"),
    (415, "Seviye B (Oklüzal-Servikal Arası)"),
    (445, "Seviye C (Servikal Altı - Derin)"),
])
def test_depth_lower(crown_y, expected):
    kp = {**M2_LOWER_LEFT, "m3_apex": (220, 540), "m3_crown": (220, crown_y)}
    assert analyze_keypoints(kp, LOWER)["depth"]["value"] == expected


@pytest.mark.parametrize("crown_y,expected", [
    (325, "Seviye A (Oklüzal)"),
    (300, "Seviye B (Oklüzal-Servikal Arası)"),
    (275, "Seviye C (Servikal Altı - Derin)"),
])
def test_depth_upper_direction_is_reversed(crown_y, expected):
    kp = {**M2_UPPER_RIGHT, "m3_apex": (980, 180), "m3_crown": (980, crown_y)}
    assert analyze_keypoints(kp, UPPER)["depth"]["value"] == expected


@pytest.mark.parametrize("ramus_x,expected", [
    (150, "Sınıf 1 (Önünde)"),          # boşluk 140 px ≥ kron genişliği 80
    (250, "Sınıf 2 (Yarı Ramus İçinde)"),
    (282, "Sınıf 3 (Tam Ramus İçinde)"),
])
def test_ramus_classes(ramus_x, expected):
    kp = {**M2_LOWER_LEFT, "ramus_anterior": (ramus_x, 400)}
    bbox = [180, 380, 260, 540]  # genişlik 80
    assert analyze_keypoints(kp, LOWER, bbox)["ramus"]["value"] == expected


def test_ramus_not_applicable_for_upper():
    assert analyze_keypoints({}, UPPER)["ramus"]["value"] == "Uygulanamaz (Üst Çene)"


@pytest.mark.parametrize("canal_y,expected", [(525, "Yakın/Temaslı"), (560, "Uzak")])
def test_nerve(canal_y, expected):
    kp = {"m3_crown": (220, 400), "m3_apex": (220, 520), "canal_top": (220, canal_y)}
    assert analyze_keypoints(kp, LOWER)["nerve"]["value"] == expected


def test_missing_points_and_partial_results():
    r = analyze_keypoints({"m3_crown": (1, 1)}, LOWER)
    assert r["impaction"] is None and r["depth"] is None and r["nerve"] is None
    assert "m3_apex" in r["missing"] and "m3_crown" not in r["missing"]


def test_upper_jaw_has_no_ramus_or_canal_points():
    assert "ramus_anterior" not in applicable_keypoints(UPPER)
    assert "canal_top" not in applicable_keypoints(UPPER)
    assert len(applicable_keypoints(LOWER)) == 8


def test_winter_falls_back_to_image_vertical_without_m2_apex():
    kp = {"m3_apex": (220, 520), "m3_crown": (220, 400), "m2_cusp_mesial": (330, 400)}
    w = winter_angle(kp, LOWER)
    assert w["value"] == "Dikey (Vertical)" and w["reference"] == "görüntü dikeyi"
