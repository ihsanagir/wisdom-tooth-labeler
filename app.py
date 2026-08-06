import io
import os
import base64
import logging
import webbrowser
import threading
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, MAX_DETECTIONS, HOST, PORT
from config import (
    GENDER_OPTIONS, AGE_OPTIONS, MOUTH_OPENING_OPTIONS,
    IMPACTION_OPTIONS, RAMUS_OPTIONS, DEPTH_OPTIONS,
    ROOT_OPTIONS, NERVE_OPTIONS,
)
from scorer import analyze_case
from image_analyzer import analyze_tooth_automatically
from post_filter import filter_wisdom_detections
from label_storage import (
    save_label as storage_save_label,
    load_label,
    get_labeled_image_stems,
    get_label_stats as storage_get_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("=== YİRMİLİK DİŞ KARAR DESTEK SİSTEMİ v1.0.2 BAŞLATILDI ===")

# Goruntu klasoru: Railway'de /data/images, lokalde train/images
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "train/images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
logger.info("Goruntu klasoru: %s", IMAGES_DIR)

model = None
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO modeli başarıyla yüklendi: %s", MODEL_PATH)
except Exception as e:
    logger.error("Model yüklenemedi: %s", e)

app = FastAPI(title="Akıllı Yirmilik Diş Karar Destek Sistemi v1.1.0", version="1.1.0")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")

# Goruntu klasoru varsa mount et
if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
    app.mount("/train-images", StaticFiles(directory=str(IMAGES_DIR)), name="train-images")
    logger.info("Goruntu klasoru mount edildi: %s", IMAGES_DIR)
else:
    logger.warning("Goruntu klasoru bos veya yok: %s", IMAGES_DIR)


class AnalyzeRequest(BaseModel):
    gender: str
    age: str
    mouth_opening: str
    impaction: str
    ramus: str
    depth: str
    root: str
    nerve: str


from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

@app.get("/")
async def serve_frontend():
    return RedirectResponse(url="/label")



@app.get("/label")
async def serve_label_frontend():
    return FileResponse("static/label.html")


@app.get("/api/options")
async def get_options():
    return {
        "gender": GENDER_OPTIONS,
        "age": AGE_OPTIONS,
        "mouth_opening": MOUTH_OPENING_OPTIONS,
        "impaction": IMPACTION_OPTIONS,
        "ramus": RAMUS_OPTIONS,
        "depth": DEPTH_OPTIONS,
        "root": ROOT_OPTIONS,
        "nerve": NERVE_OPTIONS,
    }


@app.post("/api/detect")
async def detect_teeth(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model yüklenemedi. Lütfen model dosyasını kontrol edin."}
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Geçersiz görüntü dosyası."}
        )

    img_h, img_w = image.shape[:2]

    results = model(image, conf=CONFIDENCE_THRESHOLD)
    annotated_image = image.copy()
    raw_detections = []

    all_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            all_boxes.append((box, float(conf)))

    all_boxes.sort(key=lambda x: x[1], reverse=True)

    # Tüm bbox'ları topla (cross-reference için)
    all_bbox_list = [box for box, _ in all_boxes]

    for i, (box, conf) in enumerate(all_boxes):
        if i >= MAX_DETECTIONS:
            break

        x1, y1, x2, y2 = map(int, box)

        auto_result = analyze_tooth_automatically(box, image, all_bboxes=all_bbox_list)

        raw_detections.append({
            "index": i + 1,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
            "auto_analysis": {
                "impaction": auto_result.get("impaction", "Dikey (Vertical)"),
                "impaction_confidence": auto_result.get("impaction_confidence", 0.5),
                "ramus": auto_result.get("ramus", "Sınıf 1 (Önünde)"),
                "ramus_confidence": auto_result.get("ramus_confidence", 0.5),
                "depth": auto_result.get("depth", "Seviye A (Oklüzal)"),
                "depth_confidence": auto_result.get("depth_confidence", 0.5),
                "angle_value": auto_result.get("angle_value", 0),
            },
        })

    filtered_detections, removed_detections = filter_wisdom_detections(
        raw_detections, img_w, img_h
    )

    if removed_detections:
        logger.info(
            "Anatomik filtre: %d tespit kaldırıldı → %s",
            len(removed_detections),
            [d.get("filter_reason", "?") for d in removed_detections]
        )

    for det in filtered_detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        if conf >= 0.75:
            color = (0, 200, 100)
        elif conf >= 0.55:
            color = (0, 180, 255)
        else:
            color = (0, 100, 255)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 1)
        label = f"{det['index']}.Dis"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(annotated_image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated_image, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    _, buffer = cv2.imencode(".jpg", annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    logger.info(
        "Tespit tamamlandı: %d ham → %d filtrelenmiş diş.",
        len(raw_detections), len(filtered_detections)
    )

    return {
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
        "detections": filtered_detections,
        "count": len(filtered_detections),
        "raw_count": len(raw_detections),
        "filtered_count": len(removed_detections),
    }


@app.post("/api/analyze")
async def analyze_tooth(request: AnalyzeRequest):
    result = analyze_case(
        request.gender,
        request.age,
        request.mouth_opening,
        request.impaction,
        request.ramus,
        request.depth,
        request.root,
        request.nerve,
    )
    return result


@app.get("/api/label/images")
async def list_label_images():
    if not IMAGES_DIR.exists():
        return {"images": [], "total": 0, "labeled_count": 0}

    labeled = get_labeled_image_stems()
    exts = {".jpg", ".jpeg", ".png"}
    images = [
        {"name": f.name, "labeled": f.stem in labeled}
        for f in sorted(IMAGES_DIR.iterdir())
        if f.suffix.lower() in exts
    ]
    return {
        "images": images,
        "total": len(images),
        "labeled_count": sum(1 for img in images if img["labeled"]),
    }


@app.get("/api/label/detect")
async def detect_for_label(image_name: str):
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        return JSONResponse(status_code=404, content={"error": "Görüntü bulunamadı."})

    image = cv2.imread(str(image_path))
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Görüntü okunamadı."})

    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model yüklenemedi."})

    img_h, img_w = image.shape[:2]
    results = model(image, conf=CONFIDENCE_THRESHOLD)
    all_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            all_boxes.append((box, float(conf)))

    all_boxes.sort(key=lambda x: x[1], reverse=True)
    all_bbox_list = [box for box, _ in all_boxes]

    detections = []
    for i, (box, conf) in enumerate(all_boxes[:MAX_DETECTIONS]):
        x1, y1, x2, y2 = map(int, box)
        auto = analyze_tooth_automatically(box, image, all_bboxes=all_bbox_list)
        detections.append({
            "index": i + 1,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
            "auto_impaction": auto.get("impaction", "Dikey (Vertical)"),
            "auto_ramus": auto.get("ramus", "Sınıf 1 (Önünde)"),
            "auto_depth": auto.get("depth", "Seviye A (Oklüzal)"),
        })

    return {
        "detections": detections,
        "image_width": img_w,
        "image_height": img_h,
    }


class LabelSaveRequest(BaseModel):
    image_name: str
    bbox_index: int
    bbox: list
    impaction: str
    ramus: str
    depth: str
    notes: str = ""


@app.post("/api/label/save")
async def save_label_endpoint(request: LabelSaveRequest):
    """Klinik etiketi JSON olarak kaydeder."""
    result = storage_save_label(
        request.image_name, request.bbox_index, request.bbox,
        request.impaction, request.ramus, request.depth, request.notes,
    )
    return result


@app.get("/api/label/existing")
async def get_existing_labels(image_name: str):
    return load_label(image_name)


@app.get("/api/label/stats")
async def label_stats():
    return storage_get_stats()


# ─── Admin: Goruntu Yukleme ───────────────────────────────────────────────────
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "wisdom2024admin")

@app.get("/admin")
@app.get("/admin/")
@app.get("/ADMIN")
@app.get("/ADMIN/")
@app.get("/Admin")
async def admin_page():
    admin_html_path = Path(__file__).parent / "static" / "admin.html"
    if not admin_html_path.exists():
        return JSONResponse(status_code=404, content={"error": "admin.html dosyası bulunamadı."})
    return FileResponse(str(admin_html_path))


@app.post("/api/admin/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    token: str = ""
):
    """Admin: Goruntu yukle (toplu). Token ile korunur."""
    if token != ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Yetkisiz erisim."})

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            errors.append(f"{f.filename}: desteklenmeyen format")
            continue
        dest = IMAGES_DIR / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append(f.filename)

    # Mount'u yenile (ilk yuklemede)
    if saved and "train-images" not in [r.name for r in app.routes if hasattr(r, 'name')]:
        from starlette.staticfiles import StaticFiles as SF
        app.mount("/train-images", SF(directory=str(IMAGES_DIR)), name="train-images")

    return {"saved": len(saved), "errors": errors, "files": saved}


@app.get("/api/admin/images")
async def list_admin_images(token: str = ""):
    """Admin: Yuklu goruntu listesi."""
    if token != ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Yetkisiz erisim."})
    exts = {".jpg", ".jpeg", ".png"}
    files = [f.name for f in IMAGES_DIR.iterdir() if f.suffix.lower() in exts] if IMAGES_DIR.exists() else []
    return {"count": len(files), "files": sorted(files)}


def open_browser():
    import time
    time.sleep(1.5)
    url = f"http://{HOST}:{PORT}"
    webbrowser.open(url)
    logger.info("Tarayıcı açıldı: %s", url)


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 55)
    print("  🦷 Akıllı Yirmilik Diş Karar Destek Sistemi")
    print(f"  Adres: http://{HOST}:{PORT}")
    print("  Durdurmak için: Ctrl+C")
    print("=" * 55 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=HOST, port=PORT)

