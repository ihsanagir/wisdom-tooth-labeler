
import torch
from ultralytics import YOLO


def main():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("GPU bulunamadi! CPU ile egitim cok yavas olacak.")
        return

    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data.yaml",

        epochs=150,
        patience=40,
        imgsz=896,
        batch=4,
        name="disprojesi5",
        project="trained_models",
        device=0,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.008,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=5.0,

        degrees=12.0,
        translate=0.15,
        scale=0.55,
        shear=0.0,
        perspective=0.0001,
        fliplr=0.5,
        flipud=0.0,


        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.35,

        mosaic=0.7,              # 0.5 -> 0.7: az veri icin mosaic daha onemli
        mixup=0.05,
        copy_paste=0.0,          # KAPALI
        erasing=0.3,             # 0.25 -> 0.3: okluzyona karsi daha direncli
        close_mosaic=25,         # 20 -> 25: son 25 epoch mosaic kapali (fine-tune)

        # --- Loss Agirliklari ---
        box=10.0,                # Hassas kutu lokalizasyonu
        cls=0.5,
        dfl=1.5,

        # --- NMS ---
        iou=0.5,

        # --- Diger ---
        amp=True,
        cache=False,
        workers=4,
        val=True,
        verbose=True,
        plots=True,
        save=True,
        exist_ok=False,
    )

    print("\nEgitim tamamlandi!")
    print("Sonuclar: trained_models/disprojesi5/")
    print("En iyi model: trained_models/disprojesi5/weights/best.pt")


if __name__ == "__main__":
    main()
