import os
import base64
import binascii
import logging
import secrets
import webbrowser
import threading
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, MAX_DETECTIONS, HOST, PORT, APP_VERSION, INFERENCE_DEVICE
from config import (
    GENDER_OPTIONS, AGE_OPTIONS, MOUTH_OPENING_OPTIONS,
    IMPACTION_OPTIONS, RAMUS_OPTIONS, DEPTH_OPTIONS,
    ROOT_OPTIONS, NERVE_OPTIONS,
)
from scorer import analyze_case, validate_inputs
from image_analyzer import analyze_tooth_automatically
from post_filter import filter_wisdom_detections
from label_storage import (
    save_label as storage_save_label,
    delete_label_box as storage_delete_label_box,
    load_label,
    get_labeled_image_stems,
    get_label_stats as storage_get_stats,
    export_yolo_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("=== YİRMİLİK DİŞ KARAR DESTEK SİSTEMİ v%s BAŞLATILDI ===", APP_VERSION)

# Goruntu klasoru: Railway'de /data/images, lokalde train/images
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "train/images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
logger.info("Goruntu klasoru: %s", IMAGES_DIR)

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024

# --- Erişim Kontrolü ---
# APP_USER + APP_PASSWORD tanımlıysa tüm site HTTP Basic Auth ile korunur.
APP_USER = os.getenv("APP_USER", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
# ADMIN_TOKEN tanımlı değilse admin işlemleri (yükleme/export) tamamen kapalıdır.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

if not (APP_USER and APP_PASSWORD):
    logger.warning("APP_USER/APP_PASSWORD tanımlı değil — site şifresiz çalışıyor (sadece lokal geliştirme için uygun).")
if not ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN tanımlı değil — admin yükleme ve veri seti export kapalı.")

model = None
try:
    if Path(MODEL_PATH).exists():
        model = YOLO(MODEL_PATH)
        logger.info("YOLO modeli başarıyla yüklendi: %s", MODEL_PATH)
    else:
        logger.warning("Model dosyası bulunamadı (%s) — sunucu modelsiz (hızlı) modda başlatılıyor.", MODEL_PATH)
except Exception as e:
    logger.error("Model yüklenemedi: %s", e)

app = FastAPI(title="Akıllı Yirmilik Diş Karar Destek Sistemi", version=APP_VERSION)


@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    if not (APP_USER and APP_PASSWORD) or request.url.path == "/health":
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            user, _, password = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
            if (secrets.compare_digest(user, APP_USER)
                    and secrets.compare_digest(password, APP_PASSWORD)):
                return await call_next(request)
        except (binascii.Error, UnicodeDecodeError):
            pass

    return Response(
        status_code=401,
        content="Yetkisiz erişim.",
        headers={"WWW-Authenticate": 'Basic realm="Yirmilik Dis", charset="UTF-8"'},
    )


def _check_admin(token: str) -> bool:
    return bool(ADMIN_TOKEN) and secrets.compare_digest(token or "", ADMIN_TOKEN)


def _safe_image_path(image_name: str):
    """IMAGES_DIR dışına çıkan (../) veya desteklenmeyen dosya adlarını reddeder."""
    if not image_name or Path(image_name).name != image_name:
        return None
    if Path(image_name).suffix.lower() not in ALLOWED_IMAGE_EXTS:
        return None
    return IMAGES_DIR / image_name


def _run_model(image):
    """YOLO çıkarımı — güvene göre azalan sırada [(xyxy, conf), ...] döndürür."""
    results = model(image, conf=CONFIDENCE_THRESHOLD, device=INFERENCE_DEVICE, verbose=False)
    all_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            all_boxes.append((box, float(conf)))
    all_boxes.sort(key=lambda x: x[1], reverse=True)
    return all_boxes


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None, "version": APP_VERSION}

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/train-images", StaticFiles(directory=str(IMAGES_DIR)), name="train-images")


class AnalyzeRequest(BaseModel):
    gender: str
    age: str
    mouth_opening: str
    impaction: str
    ramus: str
    depth: str
    root: str
    nerve: str


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


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


# Model çıkarımı CPU'yu bloke ettiği için bu endpoint'ler bilerek `def` (async değil):
# FastAPI bunları thread pool'da çalıştırır ve event loop kilitlenmez.
@app.post("/api/detect")
def detect_teeth(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model yüklenemedi. Lütfen model dosyasını kontrol edin."}
        )

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": "Dosya çok büyük."})

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Geçersiz görüntü dosyası (JPG veya PNG yükleyin)."}
        )

    img_h, img_w = image.shape[:2]
    all_boxes = _run_model(image)
    all_bbox_list = [box for box, _ in all_boxes]

    raw_detections = [
        {
            "index": i + 1,
            "confidence": round(conf, 4),
            "bbox": list(map(int, box)),
            "_rank": i,
        }
        for i, (box, conf) in enumerate(all_boxes)
    ]

    # Önce anatomik filtre, sonra limit: aksi halde filtre bir tespiti atınca
    # sıradaki gerçek diş listeye hiç giremiyordu.
    filtered_detections, removed_detections = filter_wisdom_detections(
        raw_detections, img_w, img_h
    )
    filtered_detections = filtered_detections[:MAX_DETECTIONS]

    if removed_detections:
        logger.info(
            "Anatomik filtre: %d tespit kaldırıldı → %s",
            len(removed_detections),
            [d.get("filter_reason", "?") for d in removed_detections]
        )

    for det in filtered_detections:
        auto_result = analyze_tooth_automatically(all_boxes[det["_rank"]][0], image, all_bboxes=all_bbox_list)
        det["auto_analysis"] = auto_result

    for det in raw_detections:
        det.pop("_rank", None)

    logger.info(
        "Tespit tamamlandı: %d ham → %d filtrelenmiş diş.",
        len(raw_detections), len(filtered_detections)
    )

    return {
        "detections": filtered_detections,
        "count": len(filtered_detections),
        "raw_count": len(raw_detections),
        "filtered_count": len(removed_detections),
    }


@app.post("/api/analyze")
async def analyze_tooth(request: AnalyzeRequest):
    invalid = validate_inputs(**request.model_dump())
    if invalid:
        return JSONResponse(
            status_code=422,
            content={"error": "Geçersiz seçenek: " + ", ".join(invalid)},
        )
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
    images = [
        {"name": f.name, "labeled": f.stem in labeled}
        for f in sorted(IMAGES_DIR.iterdir())
        if f.suffix.lower() in ALLOWED_IMAGE_EXTS
    ]
    return {
        "images": images,
        "total": len(images),
        "labeled_count": sum(1 for img in images if img["labeled"]),
    }


@app.get("/api/label/detect")
def detect_for_label(image_name: str):
    image_path = _safe_image_path(image_name)
    if image_path is None or not image_path.exists():
        return JSONResponse(status_code=404, content={"error": "Görüntü bulunamadı."})

    image = cv2.imread(str(image_path))
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Görüntü okunamadı."})

    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model yüklenemedi."})

    img_h, img_w = image.shape[:2]
    all_boxes = _run_model(image)
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
            "auto_fdi": auto.get("fdi"),
            "auto_jaw": auto.get("jaw"),
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
    root: str = "Normal/Konik"
    nerve: str = "Uzak"
    notes: str = ""


@app.post("/api/label/save")
async def save_label_endpoint(request: LabelSaveRequest):
    """Klinik etiketi JSON olarak kaydeder."""
    if _safe_image_path(request.image_name) is None:
        return JSONResponse(status_code=400, content={"error": "Geçersiz görüntü adı."})
    result = storage_save_label(
        request.image_name, request.bbox_index, request.bbox,
        request.impaction, request.ramus, request.depth,
        request.root, request.nerve, request.notes,
    )
    return result


class DeleteBoxRequest(BaseModel):
    image_name: str
    bbox_index: int


@app.post("/api/label/delete_box")
async def delete_box_endpoint(request: DeleteBoxRequest):
    """Görüntüdeki bir bbox etiketini siler."""
    if _safe_image_path(request.image_name) is None:
        return JSONResponse(status_code=400, content={"error": "Geçersiz görüntü adı."})
    return storage_delete_label_box(request.image_name, request.bbox_index)


@app.get("/api/label/export")
def export_dataset_endpoint(x_admin_token: str = Header(default="")):
    """Tüm etiketlenmiş verileri YOLO formatında ZIP olarak indirir (admin)."""
    if not _check_admin(x_admin_token):
        return JSONResponse(status_code=401, content={"error": "Yetkisiz erisim."})
    zip_buffer = export_yolo_dataset(IMAGES_DIR)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=wisdom_teeth_yolo_dataset.zip"}
    )


@app.get("/api/label/existing")
async def get_existing_labels(image_name: str):
    return load_label(image_name)


@app.get("/api/label/stats")
async def label_stats():
    return storage_get_stats()


# ─── Admin: Goruntu Yukleme ───────────────────────────────────────────────────

@app.get("/admin")
async def admin_page():
    admin_html_path = Path(__file__).parent / "static" / "admin.html"
    if not admin_html_path.exists():
        return JSONResponse(status_code=404, content={"error": "admin.html dosyası bulunamadı."})
    return FileResponse(str(admin_html_path))


@app.post("/api/admin/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    x_admin_token: str = Header(default=""),
):
    """Admin: Goruntu yukle (toplu). Token ile korunur."""
    if not _check_admin(x_admin_token):
        return JSONResponse(status_code=401, content={"error": "Yetkisiz erisim."})

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        # Tarayıcının gönderdiği yol kısımlarını at: "../x.jpg" → "x.jpg"
        name = Path(f.filename.replace("\\", "/")).name
        if Path(name).suffix.lower() not in ALLOWED_IMAGE_EXTS:
            errors.append(f"{name}: desteklenmeyen format")
            continue
        content = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            errors.append(f"{name}: dosya çok büyük")
            continue
        if cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE) is None:
            errors.append(f"{name}: geçersiz görüntü")
            continue
        (IMAGES_DIR / name).write_bytes(content)
        saved.append(name)

    return {"saved": len(saved), "errors": errors, "files": saved}


@app.get("/api/admin/images")
async def list_admin_images(x_admin_token: str = Header(default="")):
    """Admin: Yuklu goruntu listesi."""
    if not _check_admin(x_admin_token):
        return JSONResponse(status_code=401, content={"error": "Yetkisiz erisim."})
    files = [f.name for f in IMAGES_DIR.iterdir() if f.suffix.lower() in ALLOWED_IMAGE_EXTS] if IMAGES_DIR.exists() else []
    return {"count": len(files), "files": sorted(files)}


def open_browser():
    import time
    time.sleep(1.5)
    url = f"http://localhost:{PORT}"
    webbrowser.open(url)
    logger.info("Tarayıcı açıldı: %s", url)


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 55)
    print("  🦷 Akıllı Yirmilik Diş Karar Destek Sistemi")
    print(f"  Adres: http://localhost:{PORT}")
    print("  Durdurmak için: Ctrl+C")
    print("=" * 55 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=HOST, port=PORT)
