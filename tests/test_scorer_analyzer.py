import math

import cv2
import numpy as np
import pytest

from image_analyzer import JAW_LOWER, JAW_UPPER, detect_jaw, detect_tooth_angle, fdi_number
from scorer import analyze_case, validate_inputs

BASE = dict(gender="Erkek", age="<20", mouth_opening="Normal (>40mm)", root="Normal/Konik", nerve="Uzak")
EASY = dict(impaction="Mesioangular", depth="Seviye A (Oklüzal)", ramus="Sınıf 1 (Önünde)")


# ── Pederson skoru ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("case,index,severity", [
    (EASY, 3, "simple"),
    (dict(impaction="Dikey (Vertical)", depth="Seviye B (Oklüzal-Servikal Arası)", ramus="Sınıf 1 (Önünde)"), 6, "surgical"),
    (dict(impaction="Distoangular", depth="Seviye C (Servikal Altı - Derin)", ramus="Sınıf 3 (Tam Ramus İçinde)"), 10, "advanced"),
])
def test_pederson_index(case, index, severity):
    r = analyze_case(**BASE, **case)
    assert r["pederson_index"] == index and r["severity"] == severity
    assert sum(b["points"] for b in r["breakdown"]) == index


def test_major_risk_escalates_one_level():
    r = analyze_case(**{**BASE, "nerve": "Yakın/Temaslı"}, **EASY)
    assert r["base_severity"] == "simple" and r["severity"] == "surgical" and r["escalated"]


def test_single_moderate_risk_does_not_escalate_but_two_do():
    one = analyze_case(**{**BASE, "age": ">30"}, **EASY)
    two = analyze_case(**{**BASE, "age": ">30", "root": "Eğri/Dilasere"}, **EASY)
    assert one["severity"] == "simple"
    assert two["severity"] == "surgical"


def test_upper_jaw_note():
    r = analyze_case(**BASE, impaction="Dikey (Vertical)", depth="Seviye A (Oklüzal)", ramus="Uygulanamaz (Üst Çene)")
    assert r["pederson_index"] == 5 and r["notes"]


def test_validate_inputs():
    assert validate_inputs(**BASE, **EASY) == []
    assert validate_inputs(**BASE, **{**EASY, "impaction": "x"}) == ["impaction: 'x'"]


# ── Sezgisel görüntü analizi (sentetik) ──────────────────────────────────────
W, H = 1200, 600


def synthetic_tooth(cx, cy, tilt, enamel_toward_top=None):
    img = np.full((H, W), 60, np.uint8)
    cv2.ellipse(img, (cx, cy), (28, 70), tilt, 0, 360, 170, -1)
    if enamel_toward_top is not None:
        t = math.radians(tilt)
        d = (math.sin(t), -math.cos(t))
        s = 1 if enamel_toward_top else -1
        cv2.circle(img, (int(cx + s * d[0] * 50), int(cy + s * d[1] * 50)), 26, 240, -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


@pytest.mark.parametrize("cx,cy,tilt,jaw,enamel,expected", [
    (200, 400, 0, JAW_LOWER, None, "Dikey (Vertical)"),
    (200, 400, 40, JAW_LOWER, None, "Mesioangular"),
    (200, 400, -40, JAW_LOWER, None, "Distoangular"),
    (1000, 400, -40, JAW_LOWER, None, "Mesioangular"),
    (1000, 400, 40, JAW_LOWER, None, "Distoangular"),
    (200, 400, 85, JAW_LOWER, None, "Yatay (Horizontal)"),
    (200, 180, -40, JAW_UPPER, None, "Mesioangular"),
    (1000, 180, 40, JAW_UPPER, None, "Mesioangular"),
    (200, 400, 40, JAW_LOWER, True, "Mesioangular"),
    (200, 400, 0, JAW_LOWER, False, "Ters (Inverted)"),      # mine altta → kron aşağı
    (200, 400, -70, JAW_LOWER, False, "Yatay (Horizontal)"),  # kron mesial-aşağı (~110°)
])
def test_tooth_angle(cx, cy, tilt, jaw, enamel, expected):
    g = synthetic_tooth(cx, cy, tilt, enamel)
    cls, *_ = detect_tooth_angle([cx - 80, cy - 80, cx + 80, cy + 80], g, W, jaw)
    assert cls == expected


def test_jaw_from_same_side_pair():
    upper, lower = [100, 100, 180, 200], [100, 300, 180, 420]
    assert detect_jaw(upper, [lower], 1000, 600)[0] == JAW_UPPER
    assert detect_jaw(lower, [upper], 1000, 600)[0] == JAW_LOWER


@pytest.mark.parametrize("bbox,jaw,fdi", [
    ([100, 0, 200, 100], JAW_UPPER, 18), ([800, 0, 900, 100], JAW_UPPER, 28),
    ([800, 0, 900, 100], JAW_LOWER, 38), ([100, 0, 200, 100], JAW_LOWER, 48),
])
def test_fdi(bbox, jaw, fdi):
    assert fdi_number(bbox, 1000, jaw) == fdi
