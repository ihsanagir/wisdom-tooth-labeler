"""
Anatomik nokta (pose) modeli eğitimi
=====================================
Etiketleme aracında işaretlenen noktalardan YOLO-pose modeli eğitir.
Model her yirmilik diş için kutu + 8 anatomik nokta tahmin eder; geometry.py
bu noktalardan Winter / Pell & Gregory sınıflarını hesaplar.

Kullanım:
    1. Yönetim panelinden "Veri setini indir (ZIP)" → wisdom_teeth_yolo_dataset.zip
    2. python train_pose.py wisdom_teeth_yolo_dataset.zip --dry-run   # sadece veri hazırla
    3. python train_pose.py wisdom_teeth_yolo_dataset.zip             # hazırla + eğit
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from geometry import KEYPOINTS
from veri_bol import IMG_EXTS, group_images, split_groups

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "dataset_pose"
YAML_PATH = BASE_DIR / "data_pose_v1.yaml"
MIN_IMAGES = 50  # bundan az etiketli görüntüyle eğitim anlamsız


def count_points(label_text):
    """Bir pose etiket dosyasındaki işaretli nokta sayısı."""
    n = 0
    for line in label_text.split("\n"):
        vals = line.split()
        n += sum(1 for v in vals[5 + 2::3] if v == "2")
    return n


def prepare(zip_path):
    tmp = Path(tempfile.mkdtemp(prefix="pose_"))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    img_dir = tmp / "dataset" / "images"
    lbl_dir = tmp / "dataset" / "labels_pose"
    if not lbl_dir.exists():
        raise SystemExit("[X] ZIP'te labels_pose/ yok — güncel etiketleme aracıyla dışa aktarın.")

    images, total_pts = [], 0
    for p in sorted(img_dir.iterdir()):
        lbl = lbl_dir / f"{p.stem}.txt"
        if p.suffix.lower() not in IMG_EXTS or not lbl.exists():
            continue
        pts = count_points(lbl.read_text())
        if pts:
            images.append(p)
            total_pts += pts
    print(f"[*] Noktası olan görüntü: {len(images)} · toplam işaretli nokta: {total_pts}")
    if len(images) < MIN_IMAGES:
        print(f"[!] Uyarı: {MIN_IMAGES}'den az görüntü — model güvenilir olmayacak, etiketlemeye devam edin.")

    splits = split_groups(group_images(images, verbose=False))
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split, groups in splits.items():
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)
        files = [p for g in groups for p in g]
        for p in files:
            shutil.copy2(p, OUT_DIR / split / "images" / p.name)
            shutil.copy2(lbl_dir / f"{p.stem}.txt", OUT_DIR / split / "labels" / f"{p.stem}.txt")
        print(f"    {split:5s}: {len(files)} görüntü")
    shutil.rmtree(tmp, ignore_errors=True)

    YAML_PATH.write_text(
        "# train_pose.py tarafından üretildi\n"
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames:\n- Wisdom\n"
        f"kpt_shape: [{len(KEYPOINTS)}, 3]\n"
        # Noktalar anatomik rollerdir (sol/sağ çifti yok) → yatay çevirmede yer değiştirmez
        f"flip_idx: {list(range(len(KEYPOINTS)))}\n"
        f"# nokta sırası: {', '.join(KEYPOINTS)}\n",
        encoding="utf-8",
    )
    print(f"[OK] Veri: {OUT_DIR} · YAML: {YAML_PATH.name}")


def train(run_name, epochs):
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("[X] GPU bulunamadı.")
    model = YOLO("yolo11s-pose.pt")
    model.train(
        data=str(YAML_PATH), epochs=epochs, patience=50, imgsz=896, batch=4,
        project="trained_models", name=run_name, device=0,
        optimizer="AdamW", lr0=0.001, cos_lr=True,
        degrees=8.0, translate=0.1, scale=0.4, fliplr=0.5, mosaic=0.5, close_mosaic=20,
        pose=12.0, kobj=1.0,
    )
    best = f"trained_models/{run_name}/weights/best.pt"
    m = YOLO(best).val(data=str(YAML_PATH), split="test", imgsz=896, batch=4,
                       project="trained_models", name=f"{run_name}_test")
    print("\n=== TEST SETI ===")
    print(f"Kutu mAP50-95  : {m.box.map:.3f}")
    print(f"Nokta mAP50-95 : {m.pose.map:.3f}")
    print(f"Model: {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="Yönetim panelinden indirilen veri seti ZIP'i")
    ap.add_argument("--dry-run", action="store_true", help="Sadece veri setini hazırla")
    ap.add_argument("--name", default="pose1")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    prepare(args.zip)
    if not args.dry_run:
        train(args.name, args.epochs)
