"""
Akıllı Yirmilik Diş Karar Destek Sistemi — False Positive (FP) Analiz Scripti

Bu script mevcut modeli validation seti üzerinde çalıştırarak:
1. Yanlış pozitif (FP) tespitleri listeler
2. FP görüntülerini rapor halinde gösterir
3. Hangi bölgelerde FP yoğunlaştığını analiz eder
4. Post-filter uygulanmış/uygulanmamış FP sayılarını karşılaştırır

Kullanım:
    python analyze_fp.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from config import MODEL_PATH, CONFIDENCE_THRESHOLD
from post_filter import filter_wisdom_detections

# --- Sabitler ---
VAL_IMAGES_DIR = "valid/images"
VAL_LABELS_DIR = "valid/labels"
IOU_THRESHOLD = 0.5  # IoU eşiği: Ground truth ile eşleşme
OUTPUT_DIR = "fp_analysis"


def load_ground_truth(label_path, img_w, img_h):
    """YOLO formatındaki label dosyasını okur ve [x1,y1,x2,y2] formatına çevirir."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append([x1, y1, x2, y2])

    return boxes


def compute_iou(box1, box2):
    """İki bbox arasındaki IoU (Intersection over Union) hesaplar."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def analyze_fp():
    """Ana analiz fonksiyonu."""
    print("=" * 60)
    print("🔍 False Positive (FP) Analiz Raporu")
    print("=" * 60)

    # Model yükle
    model = YOLO(MODEL_PATH)
    print(f"\n📦 Model: {MODEL_PATH}")
    print(f"📂 Validation seti: {VAL_IMAGES_DIR}")
    print(f"🎯 Güven eşiği: {CONFIDENCE_THRESHOLD}")

    # Çıktı klasörü
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # İstatistikler
    total_images = 0
    total_gt = 0
    total_pred_raw = 0
    total_pred_filtered = 0
    total_tp = 0
    total_fp_raw = 0
    total_fp_filtered = 0
    total_fn = 0
    fp_positions = []  # FP tespitlerin normalize x pozisyonları
    fp_confidences = []  # FP tespitlerin güvenilirlikleri
    fp_images = []  # FP içeren görüntü isimleri

    # Validation görüntülerini tara
    val_images = sorted(Path(VAL_IMAGES_DIR).glob("*.*"))
    print(f"\n📊 {len(val_images)} görüntü analiz ediliyor...\n")

    for img_path in val_images:
        total_images += 1
        img_name = img_path.stem

        # Label dosyasını bul
        label_path = os.path.join(VAL_LABELS_DIR, img_name + ".txt")

        # Görüntüyü oku
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # Ground truth
        gt_boxes = load_ground_truth(label_path, img_w, img_h)
        total_gt += len(gt_boxes)

        # Model tahmini
        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

        pred_boxes = []
        pred_confs = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                pred_boxes.append(list(map(int, box)))
                pred_confs.append(float(conf))

        total_pred_raw += len(pred_boxes)

        # --- FP Analizi (filtresiz) ---
        gt_matched = set()
        image_fps_raw = []

        for i, (pbox, pconf) in enumerate(zip(pred_boxes, pred_confs)):
            matched = False
            for j, gbox in enumerate(gt_boxes):
                if j in gt_matched:
                    continue
                if compute_iou(pbox, gbox) >= IOU_THRESHOLD:
                    gt_matched.add(j)
                    matched = True
                    total_tp += 1
                    break

            if not matched:
                total_fp_raw += 1
                rel_x = ((pbox[0] + pbox[2]) / 2) / img_w
                fp_positions.append(rel_x)
                fp_confidences.append(pconf)
                image_fps_raw.append({
                    "bbox": pbox,
                    "confidence": pconf,
                    "rel_x": round(rel_x, 3),
                })

        # FN: Eşleşmeyen GT'ler
        total_fn += len(gt_boxes) - len(gt_matched)

        # --- Post-filter ile FP ---
        raw_dets = [
            {"index": i + 1, "confidence": c, "bbox": b}
            for i, (b, c) in enumerate(zip(pred_boxes, pred_confs))
        ]
        filtered_dets, removed_dets = filter_wisdom_detections(raw_dets, img_w, img_h)
        total_pred_filtered += len(filtered_dets)

        # Filtrelenmiş tespitlerde FP sayısı
        gt_matched_f = set()
        fp_in_filtered = 0
        for det in filtered_dets:
            matched = False
            for j, gbox in enumerate(gt_boxes):
                if j in gt_matched_f:
                    continue
                if compute_iou(det["bbox"], gbox) >= IOU_THRESHOLD:
                    gt_matched_f.add(j)
                    matched = True
                    break
            if not matched:
                fp_in_filtered += 1

        total_fp_filtered += fp_in_filtered

        if image_fps_raw:
            fp_images.append({
                "name": img_path.name,
                "fp_count_raw": len(image_fps_raw),
                "fp_count_filtered": fp_in_filtered,
                "fps": image_fps_raw,
            })

    # --- Rapor ---
    print("\n" + "=" * 60)
    print("📊 SONUÇLAR")
    print("=" * 60)

    print(f"\n📷 Toplam görüntü: {total_images}")
    print(f"🎯 Toplam ground truth: {total_gt}")

    print(f"\n--- Filtresiz (Raw) ---")
    print(f"  Toplam tahmin: {total_pred_raw}")
    print(f"  ✅ True Positive: {total_tp}")
    print(f"  ❌ False Positive: {total_fp_raw}")
    print(f"  ⬜ False Negative: {total_fn}")
    precision_raw = total_tp / (total_tp + total_fp_raw) if (total_tp + total_fp_raw) > 0 else 0
    print(f"  📈 Precision (raw): {precision_raw:.4f}")

    print(f"\n--- Post-Filter Sonrası ---")
    print(f"  Toplam tahmin: {total_pred_filtered}")
    print(f"  ❌ False Positive: {total_fp_filtered}")
    fp_reduction = ((total_fp_raw - total_fp_filtered) / total_fp_raw * 100) if total_fp_raw > 0 else 0
    precision_filt = total_tp / (total_tp + total_fp_filtered) if (total_tp + total_fp_filtered) > 0 else 0
    print(f"  📈 Precision (filtered): {precision_filt:.4f}")
    print(f"  📉 FP azaltma: %{fp_reduction:.1f}")

    # FP Pozisyon Analizi
    if fp_positions:
        print(f"\n--- FP Pozisyon Dağılımı ---")
        zones = {"Sol (%0-25)": 0, "Sol-orta (%25-50)": 0, "Sağ-orta (%50-75)": 0, "Sağ (%75-100)": 0}
        for pos in fp_positions:
            if pos < 0.25:
                zones["Sol (%0-25)"] += 1
            elif pos < 0.50:
                zones["Sol-orta (%25-50)"] += 1
            elif pos < 0.75:
                zones["Sağ-orta (%50-75)"] += 1
            else:
                zones["Sağ (%75-100)"] += 1

        for zone, count in zones.items():
            bar = "█" * count + "░" * (max(0, 20 - count))
            print(f"  {zone}: {bar} {count}")

        print(f"\n--- FP Güvenilirlik Dağılımı ---")
        conf_ranges = {"<%40": 0, "%40-55": 0, "%55-70": 0, "%70-85": 0, ">%85": 0}
        for conf in fp_confidences:
            if conf < 0.40:
                conf_ranges["<%40"] += 1
            elif conf < 0.55:
                conf_ranges["%40-55"] += 1
            elif conf < 0.70:
                conf_ranges["%55-70"] += 1
            elif conf < 0.85:
                conf_ranges["%70-85"] += 1
            else:
                conf_ranges[">%85"] += 1

        for range_name, count in conf_ranges.items():
            bar = "█" * count + "░" * (max(0, 20 - count))
            print(f"  {range_name}: {bar} {count}")

    # En çok FP içeren görüntüler
    if fp_images:
        fp_images.sort(key=lambda x: x["fp_count_raw"], reverse=True)
        print(f"\n--- En Çok FP İçeren Görüntüler (Top 10) ---")
        for item in fp_images[:10]:
            print(f"  📷 {item['name']}: {item['fp_count_raw']} FP (filtrelenmiş: {item['fp_count_filtered']})")
            for fp in item["fps"]:
                print(f"     → conf={fp['confidence']:.2f}, x={fp['rel_x']:.2f}")

    # JSON rapor kaydet
    report = {
        "total_images": total_images,
        "total_gt": total_gt,
        "total_pred_raw": total_pred_raw,
        "total_tp": total_tp,
        "total_fp_raw": total_fp_raw,
        "total_fp_filtered": total_fp_filtered,
        "total_fn": total_fn,
        "precision_raw": round(precision_raw, 4),
        "precision_filtered": round(precision_filt, 4),
        "fp_reduction_pct": round(fp_reduction, 1),
        "fp_images": fp_images[:20],
    }
    report_path = os.path.join(OUTPUT_DIR, "fp_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Rapor kaydedildi: {report_path}")

    print("\n" + "=" * 60)
    print("✅ Analiz tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    analyze_fp()
