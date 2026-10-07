"""
Sızıntısız (grup bazlı) veri seti bölme scripti
================================================
Eski bölmede aynı röntgenin Roboflow kopyaları (xxx_jpg.rf.<hash>) ve aynı hastanın
farklı görüntüleri hem train hem valid/test'e düşüyordu → metrikler şişiyordu.

Bu script:
1. Kaynak görüntüleri gruplar:
   - aynı kaynak dosya adı (.rf. öncesi)       → aynı grup
   - hasta öneki (04269002-I4 → 04269002, AC123-D → AC123, 12B → 12) → aynı grup
   - görsel olarak neredeyse aynı (64x32 küçültülmüş görüntü korelasyonu ≥ 0.945) → aynı grup
     (eşik elle doğrulandı: ≥0.947 çiftler aynı röntgen, ~0.93 çiftler farklı hasta)
2. Etiketleri doğrular; poligon satırlarını bbox'a çevirir.
3. Grupları train/valid/test'e böler (bir grup asla iki bölmeye düşmez).
4. Valid/test'te her kaynak görüntüden yalnızca 1 kopya tutar.
5. Sonucu dataset_v2/ klasörüne ve data_v2.yaml'a yazar (mevcut klasörlere dokunmaz).

Kullanım:
    python veri_bol.py
"""

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # kalan %15 test
DUPLICATE_CORR = 0.945

BASE_DIR = Path(__file__).parent
SRC_IMG = BASE_DIR / "backup_original_split" / "valid" / "images"
SRC_LBL = BASE_DIR / "backup_original_split" / "valid" / "labels"
OUT_DIR = BASE_DIR / "dataset_v2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_name(filename: str) -> str:
    """'10006002_jpg.rf.3951....jpg' → '10006002'"""
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(filename).stem)
    return re.sub(r"_(jpe?g|png)$", "", stem, flags=re.I)


def patient_key(src: str) -> str:
    """Aynı hastaya ait olduğu düşünülen görüntüler için ortak anahtar."""
    m = re.match(r"^(\d{6,})-I\d+$", src)           # 04269002-I4
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z]{2}\d+)-[A-Z]$", src)      # AC123-D
    if m:
        return m.group(1)
    m = re.match(r"^(\d{1,4})[A-Z]$", src)           # 12B
    if m:
        return m.group(1)
    return src


def thumbnail_vector(path: Path):
    """Korelasyon için normalize edilmiş 64x32 küçük görüntü vektörü."""
    # cv2.imread Windows'ta Türkçe karakterli yolları açamıyor → imdecode
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    v = cv2.resize(img, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def clean_label(lbl_path: Path):
    """YOLO etiketini okur; poligonları bbox'a çevirir, bozuk satırları atar."""
    lines, notes = [], []
    if not lbl_path.exists():
        return lines, ["label yok"]
    for raw in lbl_path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        vals = [float(p) for p in parts[1:]]
        if len(vals) == 4:
            xc, yc, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            notes.append("poligon → bbox")
        else:
            notes.append(f"bozuk satır atlandı: {raw[:40]}")
            continue
        if w <= 0 or h <= 0 or not all(0 <= v <= 1 for v in (xc, yc, w, h)):
            notes.append(f"aralık dışı satır atlandı: {raw[:40]}")
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, notes


def main():
    images = sorted(p for p in SRC_IMG.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[*] Kaynak görüntü: {len(images)}")

    uf = UnionFind()
    by_patient = defaultdict(list)
    for p in images:
        uf.find(p.name)
        by_patient[patient_key(source_name(p.name))].append(p.name)
    for names in by_patient.values():
        for n in names[1:]:
            uf.union(names[0], n)

    print("[*] Görsel benzerlik hesaplanıyor...")
    vectors = {p.name: thumbnail_vector(p) for p in images}
    names = [n for n, v in vectors.items() if v is not None]
    mat = np.array([vectors[n] for n in names])
    corr = mat @ mat.T
    near_dup = 0
    for i in range(len(names)):
        for j in np.where(corr[i, i + 1:] >= DUPLICATE_CORR)[0]:
            a, b = names[i], names[i + 1 + j]
            if uf.find(a) != uf.find(b):
                near_dup += 1
            uf.union(a, b)
    print(f"[*] Dosya adından bağımsız {near_dup} görsel kopya birleştirildi.")

    groups = defaultdict(list)
    for p in images:
        groups[uf.find(p.name)].append(p)
    group_list = sorted(groups.values(), key=lambda g: g[0].name)
    print(f"[*] Bağımsız grup sayısı: {len(group_list)}")

    random.seed(SEED)
    random.shuffle(group_list)
    total = len(images)
    splits = {"train": [], "valid": [], "test": []}
    count = 0
    for g in group_list:
        if count < total * TRAIN_RATIO:
            splits["train"].append(g)
        elif count < total * (TRAIN_RATIO + VAL_RATIO):
            splits["valid"].append(g)
        else:
            splits["test"].append(g)
        count += len(g)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    report = []
    for split, split_groups in splits.items():
        img_dir = OUT_DIR / split / "images"
        lbl_dir = OUT_DIR / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n_img = n_box = 0
        for g in split_groups:
            files = g
            if split != "train":
                # Değerlendirmede aynı kaynağın birden çok kopyası sonucu çarpıtır
                seen, files = set(), []
                for p in g:
                    if source_name(p.name) not in seen:
                        seen.add(source_name(p.name))
                        files.append(p)
            for p in files:
                lines, notes = clean_label(SRC_LBL / f"{p.stem}.txt")
                for note in notes:
                    report.append(f"{p.name}: {note}")
                shutil.copy2(p, img_dir / p.name)
                (lbl_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_box += len(lines)
        print(f"    {split:5s}: {len(split_groups):4d} grup, {n_img:4d} görüntü, {n_box:4d} kutu")

    (BASE_DIR / "data_v2.yaml").write_text(
        "# veri_bol.py tarafından üretildi — grup bazlı, sızıntısız bölme\n"
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "- Wisdom\n",
        encoding="utf-8",
    )

    if report:
        print("\n[!] Etiket düzeltmeleri:")
        for r in report:
            print("    " + r)
    print(f"\n[OK] Çıktı: {OUT_DIR}  |  YAML: data_v2.yaml")


if __name__ == "__main__":
    main()
