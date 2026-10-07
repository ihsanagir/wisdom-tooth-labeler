
import torch
from ultralytics import YOLO

# veri_bol.py ile üretilen sızıntısız (grup bazlı) bölme
DATA_YAML = "data_v2.yaml"
RUN_NAME = "disprojesi6"


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
        data=DATA_YAML,

        epochs=150,
        patience=40,
        imgsz=896,
        batch=4,
        name=RUN_NAME,
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

    best_path = f"trained_models/{RUN_NAME}/weights/best.pt"
    print("\nEgitim tamamlandi!")
    print(f"En iyi model: {best_path}")

    # Model seçiminde hiç kullanılmamış test setinde nihai, dürüst ölçüm
    test_metrics = YOLO(best_path).val(
        data=DATA_YAML, split="test", imgsz=896, batch=4,
        project="trained_models", name=f"{RUN_NAME}_test",
    )
    print("\n=== TEST SETI SONUCLARI ===")
    print(f"Precision : {test_metrics.box.mp:.3f}")
    print(f"Recall    : {test_metrics.box.mr:.3f}")
    print(f"mAP50     : {test_metrics.box.map50:.3f}")
    print(f"mAP50-95  : {test_metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
