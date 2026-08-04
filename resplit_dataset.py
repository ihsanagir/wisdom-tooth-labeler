import sys
import shutil
import random
from pathlib import Path

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20

BASE_DIR = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kaynak: sadece valid klasoru kullanilacak (train ile birebir ayni, valid 1 fazla unique)
# Duplicate'leri toplamak yerine sadece valid'den al
SOURCE_SPLIT = "backup_original_split/valid"


def main():
    print("=" * 60)
    print("  Wisdom Teeth -- Duplicate-Aware 70/20/10 Split Scripti")
    print("=" * 60)
    print()
    print("[!] TESPIT: Train ve Valid setleri birebir ayni gorselleri iceriyor.")
    print("    Gercek unique goruntu sayisi: 734")
    print("    Kaynak olarak yalnizca 'valid' klasoru kullanilacak.")
    print()

    src_img_dir = BASE_DIR / "backup_original_split" / "valid" / "images"
    src_lbl_dir = BASE_DIR / "backup_original_split" / "valid" / "labels"

    if not src_img_dir.exists():
        print(f"[X] Kaynak klasor bulunamadi: {src_img_dir}")
        print("    Once resplit_dataset.py calistirarak yedegi olusturun.")
        return

    # --- 1. Tum unique ciftleri topla ---
    pairs = []
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path if lbl_path.exists() else None))

    total = len(pairs)
    print(f"[*] Kaynak unique goruntu: {total}")

    # --- 2. Karistir ---
    random.seed(SEED)
    random.shuffle(pairs)

    # --- 3. Bol ---
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    splits_data = {
        "train": pairs[:n_train],
        "valid": pairs[n_train : n_train + n_val],
        "test" : pairs[n_train + n_val :],
    }

    print(f"\n[*] Hedeflenen Dagilim (duplicate'siz):")
    for name, sp in splits_data.items():
        print(f"    {name:5s} : {len(sp):4d}  ({len(sp)/total*100:.1f}%)")
    print()

    confirm = input("[!] train/valid/test klasorleri temizlenip yeniden doldurulacak.\n"
                    "    Devam etmek istiyor musunuz? (evet/hayir): ").strip().lower()
    if confirm not in ("evet", "e", "yes", "y"):
        print("[X] Iptal edildi.")
        return

    # --- 4. Temizle ---
    print("\n[*] Klasorler temizleniyor...")
    for split in ["train", "valid", "test"]:
        for sub in ["images", "labels"]:
            d = BASE_DIR / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        cache = BASE_DIR / split / "labels.cache"
        if cache.exists():
            cache.unlink()

    # --- 5. Kopyala ---
    print("[*] Veriler kopyalaniyor...")
    for split_name, split_pairs in splits_data.items():
        img_dir = BASE_DIR / split_name / "images"
        lbl_dir = BASE_DIR / split_name / "labels"
        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            if lbl_src and lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dir / f"{lbl_src.stem}.txt")

    # --- 6. Dogrula ---
    print("\n[OK] Tamamlandi! Sonuc:")
    total_copied = 0
    for split in ["train", "valid", "test"]:
        img_count = len(list((BASE_DIR / split / "images").iterdir()))
        lbl_count = len(list((BASE_DIR / split / "labels").iterdir()))
        total_copied += img_count
        print(f"    {split:5s} -> {img_count:4d} goruntu, {lbl_count:4d} label")
    print(f"    TOPLAM: {total_copied} goruntu (duplicate yok)")
    print()
    print("[*] Orijinal yedek: backup_original_split/ klasorunde guvende.")
    print("[OK] Model egitimi icin hazir!")


if __name__ == "__main__":
    main()
