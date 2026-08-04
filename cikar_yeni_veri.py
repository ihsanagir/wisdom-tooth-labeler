"""
Yeni Eklenen Veriyi Veri Setinden Cikart
=========================================
annotated_images.zip icindeki 93 goruntuyu backup_original_split/valid klasoru ve
train/valid/test klasorlerinden kaldirir, ardindan kalan veriyle yeniden 70/20/10 split yapar.
"""

import zipfile
import shutil
import random
from pathlib import Path

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_DIR    = Path(__file__).parent
ZIP_PATH    = BASE_DIR / "annotated_images.zip"
BACKUP_IMG  = BASE_DIR / "backup_original_split" / "valid" / "images"
BACKUP_LBL  = BASE_DIR / "backup_original_split" / "valid" / "labels"


def get_zip_filenames():
    """ZIP icindeki goruntu dosya isimlerini dondur."""
    if not ZIP_PATH.exists():
        print(f"[X] {ZIP_PATH.name} bulunamadi!")
        return set()
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        names = {Path(n).name for n in z.namelist()
                 if n.startswith("images/") and not n.endswith("/")}
    print(f"[*] ZIP'te {len(names)} goruntu tanimlandi.")
    return names


def remove_from_backup(to_remove: set):
    """Backup klasoründen goruntu ve labellarini sil."""
    removed = 0
    not_found = 0
    for fname in sorted(to_remove):
        img_path = BACKUP_IMG / fname
        stem = Path(fname).stem
        lbl_path = BACKUP_LBL / f"{stem}.txt"

        if img_path.exists():
            img_path.unlink()
            removed += 1
        else:
            print(f"  [!] Backup'ta bulunamadi: {fname}")
            not_found += 1

        if lbl_path.exists():
            lbl_path.unlink()

    print(f"[OK] Backup'tan {removed} goruntu kaldirildi. ({not_found} bulunamadi)")
    return removed


def remove_from_splits(to_remove: set):
    """train/valid/test klasorlerinden de gorselleri sil."""
    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = BASE_DIR / split / "images"
        lbl_dir = BASE_DIR / split / "labels"
        for fname in to_remove:
            img_path = img_dir / fname
            lbl_path = lbl_dir / f"{Path(fname).stem}.txt"
            if img_path.exists():
                img_path.unlink()
                total += 1
            if lbl_path.exists():
                lbl_path.unlink()
    print(f"[OK] Split klasorlerinden {total} goruntu kaldirildi.")


def resplit():
    """Backup'taki kalan veriyi 70/20/10 olarak yeniden bol."""
    pairs = []
    for img_path in sorted(BACKUP_IMG.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = BACKUP_LBL / f"{img_path.stem}.txt"
        pairs.append((img_path, lbl_path if lbl_path.exists() else None))

    total = len(pairs)
    random.seed(SEED)
    random.shuffle(pairs)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    splits_data = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_val],
        "test" : pairs[n_train + n_val:],
    }

    print(f"\n[*] Yeni dagilim ({total} goruntu):")
    for name, sp in splits_data.items():
        print(f"    {name:5s}: {len(sp):4d}  ({len(sp)/total*100:.1f}%)")

    # Klasorleri temizle ve yeniden olustur
    for split in ["train", "valid", "test"]:
        for sub in ["images", "labels"]:
            d = BASE_DIR / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        cache = BASE_DIR / split / "labels.cache"
        if cache.exists():
            cache.unlink()

    # Kopyala
    for split_name, split_pairs in splits_data.items():
        img_dir = BASE_DIR / split_name / "images"
        lbl_dir = BASE_DIR / split_name / "labels"
        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            if lbl_src and lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dir / f"{lbl_src.stem}.txt")

    print("\n[OK] Split tamamlandi:")
    for split in ["train", "valid", "test"]:
        ic = len(list((BASE_DIR / split / "images").iterdir()))
        lc = len(list((BASE_DIR / split / "labels").iterdir()))
        print(f"    {split:5s}: {ic} goruntu, {lc} label")


def main():
    print("=" * 60)
    print("  Yeni Veriyi Cikarma Scripti")
    print("=" * 60 + "\n")

    # 1. ZIP'teki dosya isimlerini al
    to_remove = get_zip_filenames()
    if not to_remove:
        print("[X] Cikarilacak dosya bulunamadi. Script sonlaniyor.")
        return

    # 2. Backup'tan sil
    print(f"\n[*] {len(to_remove)} goruntu backup'tan kaldiriliyor...")
    remove_from_backup(to_remove)

    # 3. Mevcut split klasorlerinden de sil
    print(f"\n[*] Split klasorlerinden kaldiriliyor...")
    remove_from_splits(to_remove)

    # 4. Yeniden split
    print(f"\n[*] Veri seti yeniden bolunuyor...")
    resplit()

    # Kalan goruntu sayisi
    remaining = sum(1 for _ in BACKUP_IMG.iterdir()
                    if _.suffix.lower() in IMG_EXTS)
    print(f"\n[OK] Tamamlandi! Kalan toplam goruntu: {remaining}")
    print("[*] Simdi 'python train.py' ile modeli yeniden egitebilirsiniz.")


if __name__ == "__main__":
    main()
