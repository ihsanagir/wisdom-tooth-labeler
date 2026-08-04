"""
Klinik Etiket Depolama Modülü
Her görüntü için ayrı bir JSON dosyası kullanılır.
Klasör: labels_clinical/
"""

import json
from pathlib import Path
from datetime import datetime

LABELS_DIR = Path("labels_clinical")


def _ensure_dir():
    LABELS_DIR.mkdir(exist_ok=True)


def _label_file(image_name: str) -> Path:
    return LABELS_DIR / f"{Path(image_name).stem}.json"


def save_label(image_name: str, bbox_index: int, bbox: list,
               impaction: str, ramus: str, depth: str, notes: str = "") -> dict:
    """
    Bir diş tespiti için klinik etiket kaydeder.

    Args:
        image_name  : Görüntü dosyası adı (örn. "Wisdom_1.jpg")
        bbox_index  : Diş sıra numarası (1-bazlı)
        bbox        : [x1, y1, x2, y2] koordinatları
        impaction   : Gömülülük açısı sınıfı
        ramus       : Ramus ilişkisi sınıfı
        depth       : Gömülülük derinliği sınıfı
        notes       : İsteğe bağlı klinik notlar

    Returns:
        Kayıt durumu sözlüğü
    """
    _ensure_dir()
    lf = _label_file(image_name)

    if lf.exists():
        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "image": image_name,
            "labels": [],
            "created_at": str(datetime.now()),
        }

    entry = {
        "bbox_index": bbox_index,
        "bbox": bbox,
        "impaction": impaction,
        "ramus": ramus,
        "depth": depth,
        "notes": notes,
        "labeled_at": str(datetime.now()),
    }

    # Aynı bbox_index varsa güncelle, yoksa ekle
    idx = next(
        (i for i, l in enumerate(data["labels"]) if l["bbox_index"] == bbox_index),
        None
    )
    if idx is not None:
        data["labels"][idx] = entry
    else:
        data["labels"].append(entry)

    data["updated_at"] = str(datetime.now())

    with open(lf, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "saved", "file": str(lf), "bbox_index": bbox_index}


def load_label(image_name: str) -> dict:
    """Bir görüntünün mevcut etiketlerini yükler."""
    lf = _label_file(image_name)
    if lf.exists():
        with open(lf, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"image": image_name, "labels": []}


def get_labeled_image_stems() -> set:
    """Etiketlenmiş görüntülerin dosya adlarını (uzantısız) döndürür."""
    if not LABELS_DIR.exists():
        return set()
    return {f.stem for f in LABELS_DIR.glob("*.json")}


def get_label_stats() -> dict:
    """Tüm etiketlerin özet istatistiklerini döndürür."""
    if not LABELS_DIR.exists():
        return {
            "total_labeled_images": 0,
            "total_labels": 0,
            "impaction_distribution": {},
            "ramus_distribution": {},
            "depth_distribution": {},
        }

    files = list(LABELS_DIR.glob("*.json"))
    total_labels = 0
    imp_dist, ram_dist, dep_dist = {}, {}, {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for label in data.get("labels", []):
            total_labels += 1
            for key, dist in [
                ("impaction", imp_dist),
                ("ramus", ram_dist),
                ("depth", dep_dist),
            ]:
                val = label.get(key, "Bilinmiyor")
                dist[val] = dist.get(val, 0) + 1

    return {
        "total_labeled_images": len(files),
        "total_labels": total_labels,
        "impaction_distribution": imp_dist,
        "ramus_distribution": ram_dist,
        "depth_distribution": dep_dist,
    }
