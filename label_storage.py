"""
Klinik Etiket Depolama ve YOLO Export Modülü
Her görüntü için ayrı bir JSON dosyası kullanılır.
Klasör: labels_clinical/
"""

import json
import zipfile
import io
import os

# Kalıcı etiket depolama dizini: Railway Volume için /data/labels_clinical
LABELS_DIR = Path(os.getenv("LABELS_DIR", "/data/labels_clinical" if Path("/data").exists() else "labels_clinical"))


def _ensure_dir():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _label_file(image_name: str) -> Path:
    return LABELS_DIR / f"{Path(image_name).stem}.json"


def save_label(image_name: str, bbox_index: int, bbox: list,
               impaction: str, ramus: str, depth: str,
               root: str = "Normal/Konik", nerve: str = "Uzak", notes: str = "") -> dict:
    """
    Bir diş tespiti için klinik etiket kaydeder.
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
        "root": root,
        "nerve": nerve,
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


def delete_label_box(image_name: str, bbox_index: int) -> dict:
    """Bir görüntüdeki seçili bbox etiketi siler."""
    lf = _label_file(image_name)
    if not lf.exists():
        return {"status": "not_found"}

    with open(lf, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [l for l in data.get("labels", []) if l["bbox_index"] != bbox_index]

    # Bbox index'lerini yeniden sırala (1, 2, 3...)
    for idx, l in enumerate(labels, start=1):
        l["bbox_index"] = idx

    data["labels"] = labels
    data["updated_at"] = str(datetime.now())

    with open(lf, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "deleted", "remaining_count": len(labels)}


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
            "root_distribution": {},
            "nerve_distribution": {},
        }

    files = list(LABELS_DIR.glob("*.json"))
    total_labels = 0
    imp_dist, ram_dist, dep_dist, root_dist, nerve_dist = {}, {}, {}, {}, {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for label in data.get("labels", []):
            total_labels += 1
            for key, dist in [
                ("impaction", imp_dist),
                ("ramus", ram_dist),
                ("depth", dep_dist),
                ("root", root_dist),
                ("nerve", nerve_dist),
            ]:
                val = label.get(key, "Bilinmiyor")
                dist[val] = dist.get(val, 0) + 1

    return {
        "total_labeled_images": len(files),
        "total_labels": total_labels,
        "impaction_distribution": imp_dist,
        "ramus_distribution": ram_dist,
        "depth_distribution": dep_dist,
        "root_distribution": root_dist,
        "nerve_distribution": nerve_dist,
    }


def export_yolo_dataset(images_dir: Path) -> io.BytesIO:
    """
    Etiketlenen tüm görüntüleri ve YOLO (.txt) etiketlerini ZIP olarak paketler.
    YOLO formatı: class_id x_center y_center width height (0-1 arasında normalize)
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # data.yaml ekle
        yaml_content = "names:\n  0: wisdom_tooth\nnc: 1\n"
        zip_file.writestr("dataset/data.yaml", yaml_content)

        if not LABELS_DIR.exists():
            zip_buffer.seek(0)
            return zip_buffer

        for json_file in LABELS_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_name = data.get("image")
            labels = data.get("labels", [])
            if not image_name or not labels:
                continue

            img_path = images_dir / image_name
            if not img_path.exists():
                continue

            # Görüntüyü ekle
            zip_file.write(img_path, f"dataset/images/{image_name}")

            # Görüntü boyutlarını al (YOLO normalize için)
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
            except Exception:
                continue

            # YOLO etiketlerini oluştur (.txt)
            yolo_lines = []
            for l in labels:
                bbox = l.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # YOLO normalize koordinatlar
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if yolo_lines:
                txt_name = f"{img_path.stem}.txt"
                zip_file.writestr(f"dataset/labels/{txt_name}", "\n".join(yolo_lines))

            # Klinik JSON detaylarını da sakla
            zip_file.writestr(f"dataset/clinical_json/{json_file.name}", json.dumps(data, ensure_ascii=False, indent=2))

    zip_buffer.seek(0)
    return zip_buffer
