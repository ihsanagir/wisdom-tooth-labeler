"""
Yeni Etiketlenmiş Veri Entegrasyon Scripti
============================================
isaretleme_uygulamasi.html'den indirilen annotated_images.zip dosyasini
alip projenin backup_original_split/valid klasorune ekler ve
resplit_dataset.py ile yeni 70/20/10 bolmesi yapar.

Kullanim:
    python entegre_yeni_veri.py  (zip dosyasini script ile ayni klasore koyun)
    veya:
    python entegre_yeni_veri.py "C:/Users/.../Downloads/annotated_images.zip"
"""

import sys
import shutil
import zipfile
import random
from pathlib import Path

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20

BASE_DIR = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_zip(args):
    """ZIP dosyasini bul: arguman, script klasoru veya Downloads."""
    if len(args) > 1:
        p = Path(args[1])
        if p.exists():
            return p
        print(f"[X] Belirtilen dosya bulunamadi: {p}")
        sys.exit(1)

    # Script ile ayni klasorde ara
    local = BASE_DIR / "annotated_images.zip"
    if local.exists():
        return local

    # Windows Downloads klasorunde ara
    downloads = Path.home() / "Downloads" / "annotated_images.zip"
    if downloads.exists():
        return downloads

    print("[X] annotated_images.zip bulunamadi!")
    print("    Cozum 1: zip dosyasini bu scriptle ayni klasore koyun")
    print("    Cozum 2: python entegre_yeni_veri.py 'C:/yol/annotated_images.zip'")
    sys.exit(1)


def extract_zip(zip_path):
    """ZIP'i gecici klasore cikart, image/label ciflerini dogrula."""
    tmp_dir = BASE_DIR / "_tmp_new_data"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    print(f"[*] ZIP aciliyor: {zip_path.name}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)

    # images/ ve labels/ klasorlerini bul
    img_dir = tmp_dir / "images"
    lbl_dir = tmp_dir / "labels"

    if not img_dir.exists():
        print("[X] ZIP icinde 'images/' klasoru bulunamadi!")
        sys.exit(1)

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"[!] Label yok, atlaniyor: {img_path.name}")
            continue
        # Bos label kontrolu
        content = lbl_path.read_text().strip()
        if not content:
            print(f"[!] Bos label, atlaniyor: {img_path.name}")
            continue
        pairs.append((img_path, lbl_path))

    print(f"[OK] {len(pairs)} gecerli goruntu/label cifti bulundu.")
    return pairs, tmp_dir


def add_to_backup(new_pairs):
    """Yeni cifrleri backup_original_split/valid klasorune ekle (kaynak havuzu)."""
    backup_valid_img = BASE_DIR / "backup_original_split" / "valid" / "images"
    backup_valid_lbl = BASE_DIR / "backup_original_split" / "valid" / "labels"

    if not backup_valid_img.exists():
        print("[X] backup_original_split/valid bulunamadi!")
        print("    Once resplit_dataset.py'yi calistirarak yedek olusturun.")
        sys.exit(1)

    added = 0
    skipped = 0
    for img_src, lbl_src in new_pairs:
        dst_img = backup_valid_img / img_src.name
        dst_lbl = backup_valid_lbl / lbl_src.name

        if dst_img.exists():
            print(f"[!] Zaten mevcut, atlaniyor: {img_src.name}")
            skipped += 1
            continue

        shutil.copy2(img_src, dst_img)
        shutil.copy2(lbl_src, dst_lbl)
        added += 1

    print(f"[OK] {added} yeni goruntu eklendi, {skipped} atlandı (zaten vardı).")
    return added


def resplit(total_backup):
    """backup/valid'deki tum veriyi 70/20/10 oraninda yeniden bol."""
    src_img_dir = BASE_DIR / "backup_original_split" / "valid" / "images"
    src_lbl_dir = BASE_DIR / "backup_original_split" / "valid" / "labels"

    pairs = []
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
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

    print(f"\n[*] Yeni dagilim ({total} unique goruntu):")
    for name, sp in splits_data.items():
        print(f"    {name:5s}: {len(sp):4d}  ({len(sp)/total*100:.1f}%)")

    # Temizle
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

    # Dogrula
    print("\n[OK] Split tamamlandi:")
    for split in ["train", "valid", "test"]:
        ic = len(list((BASE_DIR / split / "images").iterdir()))
        lc = len(list((BASE_DIR / split / "labels").iterdir()))
        print(f"    {split:5s}: {ic} goruntu, {lc} label")


def main():
    print("=" * 60)
    print("  Yeni Veri Entegrasyon Scripti")
    print("=" * 60 + "\n")

    # 1. ZIP bul
    zip_path = find_zip(sys.argv)
    print(f"[*] ZIP bulundu: {zip_path}\n")

    # 2. ZIP'i ac ve dogrula
    new_pairs, tmp_dir = extract_zip(zip_path)
    if not new_pairs:
        print("[X] Eklenecek gecerli veri yok.")
        shutil.rmtree(tmp_dir)
        return

    # 3. Backup'a ekle
    print(f"\n[*] Yeni veriler backup havuzuna ekleniyor...")
    added = add_to_backup(new_pairs)

    # 4. Temizlik
    shutil.rmtree(tmp_dir)

    if added == 0:
        print("[!] Hic yeni veri eklenmedi. Script sonlaniyor.")
        return

    # 5. Yeniden bol
    print(f"\n[*] Veri seti yeniden bolunuyor...")
    resplit(added)

    print("\n[OK] Tum islemler tamamlandi!")
    print("[*] Simdi 'python train.py' ile modeli yeniden egitebilirsiniz.")


if __name__ == "__main__":
    main()
