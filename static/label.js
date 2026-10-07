/**
 * Klinik Etiketleme Arayüzü — JavaScript (BBox Çizimi + Klinik Etiketleme + YOLO Export)
 */

// ============================================================
// SABITLER
// ============================================================

const IMPACTION_OPTIONS = [
    "Dikey (Vertical)",
    "Mesioangular",
    "Distoangular",
    "Yatay (Horizontal)",
    "Ters (Inverted)",
];
const RAMUS_OPTIONS = [
    "Sınıf 1 (Önünde)",
    "Sınıf 2 (Yarı Ramus İçinde)",
    "Sınıf 3 (Tam Ramus İçinde)",
    "Uygulanamaz (Üst Çene)",
];
const DEPTH_OPTIONS = [
    "Seviye A (Oklüzal)",
    "Seviye B (Oklüzal-Servikal Arası)",
    "Seviye C (Servikal Altı - Derin)",
];
const ROOT_OPTIONS = [
    "Normal/Konik",
    "Eğri/Dilasere",
    "Ayrık/Diverjan",
];
const NERVE_OPTIONS = [
    "Uzak",
    "Yakın/Temaslı",
];

// ============================================================
// DURUM (STATE)
// ============================================================

let allImages       = [];   // {name, labeled}
let currentFilter   = "all";
let currentImage    = null; // seçili görüntü adı
let detections      = [];   // [{index, bbox: [x1, y1, x2, y2], confidence, auto_impaction, ...}]
let savedLabels     = {};   // bbox_index → {impaction, ramus, depth, root, nerve, notes}
let selectedIdx     = null; // aktif diş index'i (1-bazlı)
let imgNaturalW     = 0;
let imgNaturalH     = 0;
let canvasScale     = 1;

// Çizim Durumu (Drawing State)
let isDrawing       = false;
let startX          = 0;
let startY          = 0;
let currentX        = 0;
let currentY        = 0;

// Image nesnesi önbellek
let currentImgObj   = null;

// ============================================================
// DOM KISAYOLLARI
// ============================================================

const $ = (id) => document.getElementById(id);

// Kanvas renkleri theme.css değişkenlerinden okunur
const css = getComputedStyle(document.documentElement);
const THEME = {
    idle:     css.getPropertyValue("--text-3").trim(),
    saved:    css.getPropertyValue("--ok").trim(),
    selected: css.getPropertyValue("--accent").trim(),
    ink:      css.getPropertyValue("--accent-ink").trim(),
    mono:     css.getPropertyValue("--mono").trim(),
    axis:     "#ff8a65",
    occlusal: "#f0cf6a",
};

// Anatomik nokta renkleri (sıra geometry.KEYPOINTS ile aynı)
const KP_COLORS = ["#ff8a65", "#ff8a65", "#f0cf6a", "#f0cf6a", "#c9a7ff", "#c9a7ff", "#67d4d0", "#7fb2ff"];
let kpSchema   = [];     // [{name, label}]
let kpUpperJaw = [];     // üst çenede geçerli nokta adları
let activeKp   = null;   // işaretlenmekte olan nokta adı
let geomTimer  = null;
let lastGeometry = null;

function jawOf(det) {
    if (det.jaw) return det.jaw;
    if (det.auto_jaw) return det.auto_jaw;
    const cy = (det.bbox[1] + det.bbox[3]) / 2;
    return cy < imgNaturalH * 0.52 ? "Üst Çene" : "Alt Çene";
}

function applicableKps(det) {
    return jawOf(det) === "Üst Çene" ? kpSchema.filter(k => kpUpperJaw.includes(k.name)) : kpSchema;
}

function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 0 ? inter / ua : 0;
}

function nextIndex() {
    return detections.reduce((m, d) => Math.max(m, d.index), 0) + 1;
}

function toothName(det) {
    if (det.auto_fdi && !det.jaw) return String(det.auto_fdi);
    if (!imgNaturalW) return `#${det.index}`;
    // FDI: görüntünün solu hastanın sağıdır (18/48), sağı hastanın solu (28/38)
    const left = (det.bbox[0] + det.bbox[2]) / 2 < imgNaturalW / 2;
    const upper = jawOf(det) === "Üst Çene";
    return String(left ? (upper ? 18 : 48) : (upper ? 28 : 38));
}
const canvas        = $("labelCanvas");
const ctx           = canvas.getContext("2d");
const canvasWrap    = $("canvasWrap");
const placeholder   = $("canvasPlaceholder");
const imageList     = $("imageList");
const toothTabs     = $("toothTabs");
const labelForm     = $("labelForm");
const noSelMsg      = $("noSelectionMsg");
const progressFill  = $("progressFill");
const progressText  = $("progressText");

// ============================================================
// BAŞLANGIÇ
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    buildRadioGroups();
    loadImageList();
    initCanvasEvents();
    loadKeypointSchema();
    initJawToggle();
    document.addEventListener("keydown", onKeyDown);
    // Elle yapılan seçimler öneri satırlarındaki "seçili" durumunu güncellesin
    labelForm.addEventListener("click", e => {
        if (e.target.closest(".radio-option")) setTimeout(renderSuggestions, 0);
    });

    $("statsToggle").addEventListener("click", showStats);
    window.addEventListener("resize", () => { if (currentImage) redrawCanvas(); });
});

// ============================================================
// GÖRÜNTÜ LİSTESİ
// ============================================================

async function loadImageList() {
    try {
        const res  = await fetch("/api/label/images");
        const data = await res.json();
        allImages  = data.images || [];
        updateProgress(data.labeled_count, data.total);
        renderImageList();
    } catch (e) {
        imageList.innerHTML = `<div class="list-loading error">Görüntüler yüklenemedi.</div>`;
    }
}

function updateProgress(labeled, total) {
    const pct = total > 0 ? Math.round((labeled / total) * 100) : 0;
    progressFill.style.width = pct + "%";
    progressText.textContent = `${labeled} / ${total} etiketlendi (%${pct})`;
}

function setFilter(filter, btn) {
    currentFilter = filter;
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderImageList();
}

function renderImageList() {
    const items = allImages.filter(img => {
        if (currentFilter === "labeled")   return img.labeled;
        if (currentFilter === "unlabeled") return !img.labeled;
        return true;
    });

    if (items.length === 0) {
        imageList.innerHTML = `<div class="list-loading">Bu filtrede görüntü yok.</div>`;
        return;
    }

    imageList.innerHTML = "";
    items.forEach(img => {
        const div = document.createElement("div");
        div.className = "img-item" + (img.name === currentImage ? " active" : "");
        const dot = document.createElement("span");
        dot.className = "img-dot " + (img.labeled ? "labeled" : "unlabeled");
        const nameEl = document.createElement("span");
        nameEl.className = "img-name";
        nameEl.title = img.name;
        nameEl.textContent = img.name;
        div.append(dot, nameEl);
        div.onclick = () => selectImage(img.name);
        imageList.appendChild(div);
    });
}

// ============================================================
// GÖRÜNTÜ SEÇİMİ & YOLO/MEVCUT TESPİTLER
// ============================================================

async function selectImage(name) {
    currentImage = name;
    selectedIdx  = null;
    detections   = [];
    savedLabels  = {};
    toggleMobileSidebar(false); // Mobilde sol çekmeceyi otomatik kapat
    renderImageList();
    showLabelForm(false);
    toothTabs.innerHTML = `<span class="hint">Yükleniyor…</span>`;

    placeholder.style.display = "none";
    canvas.style.display = "block";

    await drawImageOnCanvas(name);

    // Mevcut etiketleri yükle
    try {
        const existing = await fetch(`/api/label/existing?image_name=${encodeURIComponent(name)}`);
        const exData   = await existing.json();
        const existingLabels = exData.labels || [];

        existingLabels.forEach(l => {
            savedLabels[l.bbox_index] = l;
            detections.push({
                index: l.bbox_index,
                bbox: l.bbox,
                confidence: 1.0,
                auto_impaction: l.impaction || "Dikey (Vertical)",
                auto_ramus: l.ramus || "Sınıf 1 (Önünde)",
                auto_depth: l.depth || "Seviye A (Oklüzal)",
                keypoints: l.keypoints || {},
                jaw: l.jaw || null,
            });
        });

        // Model tespiti dene (eğer model yüklüyse)
        try {
            const res  = await fetch(`/api/label/detect?image_name=${encodeURIComponent(name)}`);
            const data = await res.json();
            if (!data.error && data.detections) {
                // Model çıktılarını ekle (eğer önceden eklenmediyse)
                // Kayıtlı kutularla örtüşen öneriler atlanır; yenilere çakışmayan indeks verilir
                data.detections.forEach(d => {
                    if (detections.some(ex => iou(ex.bbox, d.bbox) > 0.5)) return;
                    detections.push({ ...d, index: nextIndex(), keypoints: {} });
                });
            }
        } catch (e) {
            // Model yüklenemedi uyarısı normal (modelsiz mod)
        }

        renderTabs();
        redrawCanvas();

        if (detections.length > 0) {
            const first = detections[0];
            selectTooth(first.index);
        } else {
            toothTabs.innerHTML = `<span class="hint">Model diş bulamadı — görüntü üzerinde sürükleyerek kutu çizin.</span>`;
        }

    } catch (e) {
        toothTabs.innerHTML = "";
        const err = document.createElement("span");
        err.className = "hint error";
        err.textContent = e.message;
        toothTabs.appendChild(err);
    }
}

async function drawImageOnCanvas(name) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            currentImgObj = img;
            imgNaturalW   = img.width;
            imgNaturalH   = img.height;

            const wrap  = canvasWrap;
            const maxW  = wrap.clientWidth;
            const maxH  = wrap.clientHeight - 4;
            const scale = Math.min(maxW / img.width, maxH / img.height);

            canvas.width  = img.width  * scale;
            canvas.height = img.height * scale;

            canvasScale   = scale;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve();
        };
        img.src = `/train-images/${encodeURIComponent(name)}`;
    });
}

function redrawCanvas() {
    if (!currentImgObj) return;
    const wrap  = canvasWrap;
    const maxW  = wrap.clientWidth;
    const maxH  = wrap.clientHeight - 4;
    const scale = Math.min(maxW / currentImgObj.width, maxH / currentImgObj.height);

    canvas.width  = currentImgObj.width  * scale;
    canvas.height = currentImgObj.height * scale;
    canvasScale   = scale;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImgObj, 0, 0, canvas.width, canvas.height);
    drawBboxes();
}

// ============================================================
// CANVAS: BBOX ÇİZİMİ & FARE ETKİNLİKLERİ
// ============================================================

function drawBboxes() {
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const sx = x1 * canvasScale;
        const sy = y1 * canvasScale;
        const sw = (x2 - x1) * canvasScale;
        const sh = (y2 - y1) * canvasScale;

        const isSelected = det.index === selectedIdx;
        const isSaved    = !!savedLabels[det.index];

        let color = THEME.idle;
        if (isSaved)    color = THEME.saved;
        if (isSelected) color = THEME.selected;

        ctx.strokeStyle = color;
        ctx.lineWidth   = isSelected ? 2.5 : 1.5;
        ctx.strokeRect(sx, sy, sw, sh);

        // Etiket kutusu
        const label = toothName(det);
        ctx.font = `500 12px ${THEME.mono}`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(sx, sy - 19, tw + 10, 18);
        ctx.fillStyle = THEME.ink;
        ctx.fillText(label, sx + 5, sy - 6);
    });

    drawKeypoints();

    // Halen çizilmekte olan geçici kutu
    if (isDrawing) {
        const x = Math.min(startX, currentX) * canvasScale;
        const y = Math.min(startY, currentY) * canvasScale;
        const w = Math.abs(currentX - startX) * canvasScale;
        const h = Math.abs(currentY - startY) * canvasScale;

        ctx.strokeStyle = THEME.selected;
        ctx.lineWidth   = 2;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        ctx.fillStyle = THEME.selected;
        ctx.font = `500 12px ${THEME.mono}`;
        ctx.fillText("yeni kutu", x + 4, y - 6);
    }
}

function initCanvasEvents() {
    // Fare Koordinatları
    const getPos = (e) => {
        const rect = canvas.getBoundingClientRect();
        let clientX = e.clientX;
        let clientY = e.clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }
        return {
            x: (clientX - rect.left) / canvasScale,
            y: (clientY - rect.top)  / canvasScale
        };
    };

    const handleStart = (e) => {
        if (!currentImage) return;
        const pos = getPos(e);
        const mx = pos.x;
        const my = pos.y;

        // Nokta işaretleme modu: tıklanan yere aktif noktayı koy
        if (activeKp && selectedIdx !== null) {
            placeKeypoint(Math.round(mx), Math.round(my));
            return;
        }

        // Önce var olan bir kutuya mı tıklandı/dokunuldu kontrol et
        for (const det of detections) {
            const [x1, y1, x2, y2] = det.bbox;
            if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
                selectTooth(det.index);
                return;
            }
        }

        // Yeni kutu çizimi başlat
        isDrawing = true;
        startX = mx;
        startY = my;
        currentX = mx;
        currentY = my;
    };

    const handleMove = (e) => {
        if (!isDrawing) return;
        if (e.cancelable) e.preventDefault(); // Mobilde kaydırmayı engelle
        const pos = getPos(e);
        currentX = pos.x;
        currentY = pos.y;
        redrawCanvas();
    };

    const handleEnd = () => {
        if (!isDrawing) return;
        isDrawing = false;

        const x1 = Math.round(Math.min(startX, currentX));
        const y1 = Math.round(Math.min(startY, currentY));
        const x2 = Math.round(Math.max(startX, currentX));
        const y2 = Math.round(Math.max(startY, currentY));

        // Çok küçük kutuları görmezden gel (min 15x15 px)
        if ((x2 - x1) < 15 || (y2 - y1) < 15) {
            redrawCanvas();
            return;
        }

        // Yeni kutuyu ekle
        const newIdx = nextIndex();
        const newDet = {
            index: newIdx,
            bbox: [x1, y1, x2, y2],
            keypoints: {},
            confidence: 1.0,
            auto_impaction: "Dikey (Vertical)",
            auto_ramus: "Sınıf 1 (Önünde)",
            auto_depth: "Seviye A (Oklüzal)",
        };

        detections.push(newDet);
        renderTabs();
        selectTooth(newIdx);
    };

    // Fare Etkinlikleri (Mouse Events)
    canvas.addEventListener("mousedown", handleStart);
    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("mouseup", handleEnd);

    // Dokunmatik Ekran Etkinlikleri (Touch Events for Mobile/Tablet)
    canvas.addEventListener("touchstart", (e) => {
        if (isDrawing && e.cancelable) e.preventDefault();
        handleStart(e);
    }, { passive: false });

    canvas.addEventListener("touchmove", handleMove, { passive: false });
    canvas.addEventListener("touchend", handleEnd);
    canvas.addEventListener("touchcancel", handleEnd);
}

function toggleMobileSidebar(forceState) {
    const sidebar = $("sidebarLeft");
    const backdrop = $("sidebarBackdrop");
    if (!sidebar) return;

    const isOpen = typeof forceState === "boolean" ? forceState : !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", isOpen);
    if (backdrop) backdrop.classList.toggle("open", isOpen);
}

function enableDrawMode() {
    const hint = document.querySelector(".draw-hint");
    hint.classList.remove("flash");
    void hint.offsetWidth;  // animasyonu yeniden başlat
    hint.classList.add("flash");
}

// ============================================================
// DİŞ SEÇİMİ VE SİLME
// ============================================================

function selectTooth(index) {
    selectedIdx = index;

    document.querySelectorAll(".tooth-tab").forEach(t => {
        t.classList.toggle("active", parseInt(t.dataset.idx) === index);
    });

    redrawCanvas();

    if (index === null) {
        showLabelForm(false);
        return;
    }

    const det = detections.find(d => d.index === index);
    if (!det) return;

    showLabelForm(true);
    $("formToothTitle").textContent = `Diş ${toothName(det)} · ${jawOf(det)}`;

    $("autoHint").innerHTML =
        `Kutu <b>x</b> ${det.bbox[0]}–${det.bbox[2]} · <b>y</b> ${det.bbox[1]}–${det.bbox[3]}` +
        (savedLabels[index] ? " · kayıtlı etiket" : ` · model güveni %${Math.round(det.confidence * 100)}`);

    if (!det.keypoints) det.keypoints = {};
    renderJawToggle(det);
    setActiveKp(null);
    renderKpList();
    lastGeometry = null;
    $("kpSuggest").innerHTML = "";
    requestGeometry();

    const saved = savedLabels[index];
    setRadio("impactionGroup", saved?.impaction ?? det.auto_impaction);
    setRadio("ramusGroup",     saved?.ramus     ?? det.auto_ramus);
    setRadio("depthGroup",     saved?.depth     ?? det.auto_depth);
    setRadio("rootGroup",      saved?.root      ?? "Normal/Konik");
    setRadio("nerveGroup",     saved?.nerve     ?? "Uzak");
    $("labelNotes").value = saved?.notes ?? "";

    $("saveFeedback").textContent = "";
}

async function deleteCurrentBox() {
    if (selectedIdx === null || !currentImage) return;

    const target = detections.find(d => d.index === selectedIdx);
    if (!confirm(`${target ? toothName(target) : selectedIdx} kutusunu ve etiketini silmek istediğinize emin misiniz?`)) return;

    try {
        await fetch("/api/label/delete_box", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_name: currentImage, bbox_index: selectedIdx })
        });

        // İndeksler sabit kimliktir (backend de yeniden numaralandırmaz)
        delete savedLabels[selectedIdx];
        detections = detections.filter(d => d.index !== selectedIdx);

        renderTabs();
        selectTooth(detections.length > 0 ? detections[0].index : null);
        redrawCanvas();
        updateImageLabeledStatus();
        showFeedback("Kutu silindi.", true);

    } catch (e) {
        showFeedback("Silme hatası: " + e.message, false);
    }
}

// ============================================================
// SEKME OLUŞTURMA
// ============================================================

function renderTabs() {
    toothTabs.innerHTML = "";
    if (detections.length === 0) {
        toothTabs.innerHTML = `<span class="hint">Görüntü üzerinde sürükleyerek diş kutusu çizin.</span>`;
        return;
    }
    detections.forEach(det => {
        const btn = document.createElement("button");
        btn.className = "tooth-tab" + (savedLabels[det.index] ? " saved" : "");
        btn.dataset.idx = det.index;
        btn.textContent = toothName(det);
        btn.onclick = () => selectTooth(det.index);
        toothTabs.appendChild(btn);
    });
}

function updateTab(index) {
    const tab = toothTabs.querySelector(`[data-idx="${index}"]`);
    if (tab) tab.classList.add("saved");
}

// ============================================================
// FORM YARDIMCILARI
// ============================================================

function buildRadioGroups() {
    buildGroup("impactionGroup", IMPACTION_OPTIONS);
    buildGroup("ramusGroup",     RAMUS_OPTIONS);
    buildGroup("depthGroup",     DEPTH_OPTIONS);
    buildGroup("rootGroup",      ROOT_OPTIONS);
    buildGroup("nerveGroup",     NERVE_OPTIONS);
}

function buildGroup(containerId, options) {
    const container = $(containerId);
    if (!container) return;
    container.innerHTML = "";
    options.forEach(opt => {
        const div = document.createElement("div");
        div.className = "radio-option";
        div.dataset.value = opt;
        div.innerHTML = `<span class="radio-dot"></span><span>${opt}</span>`;
        div.onclick = () => setRadio(containerId, opt);
        container.appendChild(div);
    });
}

function setRadio(containerId, value) {
    document.querySelectorAll(`#${containerId} .radio-option`).forEach(el => {
        el.classList.toggle("selected", el.dataset.value === value);
    });
}

function getRadio(containerId) {
    const sel = document.querySelector(`#${containerId} .radio-option.selected`);
    return sel ? sel.dataset.value : null;
}

function showLabelForm(show) {
    labelForm.style.display   = show ? "block" : "none";
    noSelMsg.style.display    = show ? "none"  : "flex";
}

function toggleGuide(id) {
    const el = $(id);
    if (el) el.classList.toggle("open");
}

// ============================================================
// KAYDETME
// ============================================================

async function saveCurrentLabel() {
    if (selectedIdx === null || !currentImage) return;

    const impaction = getRadio("impactionGroup");
    const ramus     = getRadio("ramusGroup");
    const depth     = getRadio("depthGroup");
    const root      = getRadio("rootGroup") || "Normal/Konik";
    const nerve     = getRadio("nerveGroup") || "Uzak";

    if (!impaction || !ramus || !depth) {
        showFeedback("Lütfen açı, ramus ve derinlik alanlarını seçin.", false);
        return;
    }

    const det = detections.find(d => d.index === selectedIdx);
    const body = {
        image_name: currentImage,
        bbox_index: selectedIdx,
        bbox:       det ? det.bbox : [],
        impaction,
        ramus,
        depth,
        root,
        nerve,
        notes: $("labelNotes").value,
        keypoints: det ? det.keypoints || {} : {},
        jaw: det ? jawOf(det) : null,
    };

    $("saveBtnText").textContent = "Kaydediliyor…";

    try {
        const res  = await fetch("/api/label/save", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        });
        const data = await res.json();

        if (data.status === "saved") {
            savedLabels[selectedIdx] = body;
            updateTab(selectedIdx);
            redrawCanvas();
            updateImageLabeledStatus();
            showFeedback("Kaydedildi.", true);

            const next = detections.find(d => d.index > selectedIdx && !savedLabels[d.index]);
            if (next) setTimeout(() => selectTooth(next.index), 600);
        } else {
            showFeedback("Kaydedilemedi.", false);
        }
    } catch (e) {
        showFeedback("Hata: " + e.message, false);
    } finally {
        $("saveBtnText").textContent = "Kaydet";
    }
}

function showFeedback(msg, ok) {
    const el = $("saveFeedback");
    el.textContent = msg;
    el.className   = "save-feedback " + (ok ? "ok" : "err");
    setTimeout(() => { el.textContent = ""; el.className = "save-feedback"; }, 3000);
}

function updateImageLabeledStatus() {
    const allSaved = detections.length > 0 && detections.every(d => savedLabels[d.index]);

    const img = allImages.find(i => i.name === currentImage);
    if (img) {
        img.labeled = allSaved;
        const labeled = allImages.filter(i => i.labeled).length;
        updateProgress(labeled, allImages.length);
        renderImageList();
    }
}

// ============================================================
// İSTATİSTİKLER
// ============================================================

async function showStats() {
    $("statsOverlay").style.display = "flex";
    $("statsContent").innerHTML = "Yükleniyor…";
    try {
        const res  = await fetch("/api/label/stats");
        const data = await res.json();
        renderStats(data);
    } catch (e) {
        $("statsContent").innerHTML = `<span class="hint error">İstatistikler yüklenemedi.</span>`;
    }
}

function closeStats() { $("statsOverlay").style.display = "none"; }

function renderStats(data) {
    const distHtml = (obj) => (!obj || Object.entries(obj).length === 0)
        ? `<div class="stat-row"><span>Henüz etiket yok</span><span>—</span></div>`
        : Object.entries(obj).map(([k, v]) =>
            `<div class="stat-row"><span>${k}</span><span>${v}</span></div>`
          ).join("");

    $("statsContent").innerHTML = `
        <div class="stat-summary">
            <div class="stat-big"><div class="num">${data.total_labeled_images}</div><div class="lbl">Görüntü</div></div>
            <div class="stat-big"><div class="num">${data.total_labels}</div><div class="lbl">Etiket</div></div>
        </div>
        <div class="stat-section"><h4>Gömülülük Açısı</h4>${distHtml(data.impaction_distribution)}</div>
        <div class="stat-section"><h4>Ramus İlişkisi</h4>${distHtml(data.ramus_distribution)}</div>
        <div class="stat-section"><h4>Gömülülük Derinliği</h4>${distHtml(data.depth_distribution)}</div>
        <div class="stat-section"><h4>Kök Morfolojisi</h4>${distHtml(data.root_distribution)}</div>
        <div class="stat-section"><h4>Sinir İlişkisi</h4>${distHtml(data.nerve_distribution)}</div>
    `;
}


// ============================================================
// ANATOMİK NOKTALAR (keypoint) — geometrik öneri + pose eğitim verisi
// ============================================================

async function loadKeypointSchema() {
    try {
        const data = await fetch("/api/label/keypoints").then(r => r.json());
        kpSchema = data.keypoints;
        kpUpperJaw = data.upper_jaw;
    } catch (e) {
        kpSchema = [];
    }
}

function currentDet() {
    return detections.find(d => d.index === selectedIdx) || null;
}

function initJawToggle() {
    $("jawToggle").querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            const det = currentDet();
            if (!det) return;
            det.jaw = btn.dataset.jaw;
            // Üst çenede geçersiz noktaları temizle
            const allowed = applicableKps(det).map(k => k.name);
            Object.keys(det.keypoints).forEach(k => { if (!allowed.includes(k)) delete det.keypoints[k]; });
            renderJawToggle(det);
            renderKpList();
            renderTabs();
            toothTabs.querySelector(`[data-idx="${det.index}"]`)?.classList.add("active");
            $("formToothTitle").textContent = `Diş ${toothName(det)} · ${jawOf(det)}`;
            redrawCanvas();
            requestGeometry();
        });
    });
}

function renderJawToggle(det) {
    const jaw = jawOf(det);
    $("jawToggle").querySelectorAll("button").forEach(b =>
        b.setAttribute("aria-checked", String(b.dataset.jaw === jaw)));
}

function renderKpList() {
    const det = currentDet();
    const list = $("kpList");
    list.innerHTML = "";
    if (!det || !kpSchema.length) return;

    applicableKps(det).forEach(k => {
        const i = kpSchema.findIndex(s => s.name === k.name);
        const pt = det.keypoints[k.name];
        const row = document.createElement("button");
        row.type = "button";
        row.className = "kp-row" + (pt ? " placed" : "") + (activeKp === k.name ? " active" : "");
        row.style.setProperty("--kp-color", KP_COLORS[i]);
        row.dataset.kp = k.name;

        const num = document.createElement("span");
        num.className = "kp-num";
        num.textContent = i + 1;
        const label = document.createElement("span");
        label.textContent = k.label;
        const state = document.createElement("span");
        state.className = "kp-state";
        state.textContent = pt ? "işaretli" : activeKp === k.name ? "tıklayın…" : "";
        const clear = document.createElement("span");
        clear.className = "kp-clear";
        clear.title = "Noktayı sil";
        clear.innerHTML = '<svg class="ico"><use href="#i-x"/></svg>';
        clear.addEventListener("click", (e) => {
            e.stopPropagation();
            delete det.keypoints[k.name];
            renderKpList();
            redrawCanvas();
            requestGeometry();
        });

        row.append(num, label, state, clear);
        row.addEventListener("click", () => setActiveKp(activeKp === k.name ? null : k.name));
        list.appendChild(row);
    });
}

function setActiveKp(name) {
    activeKp = name;
    canvas.classList.toggle("kp-mode", !!name);
    renderKpList();
}

function placeKeypoint(x, y) {
    const det = currentDet();
    if (!det || !activeKp) return;
    det.keypoints[activeKp] = [x, y];
    // Sıradaki eksik noktaya geç
    const next = applicableKps(det).find(k => !det.keypoints[k.name]);
    setActiveKp(next ? next.name : null);
    redrawCanvas();
    requestGeometry();
}

function onKeyDown(e) {
    if (e.target.closest?.("input, textarea, select")) return;
    const det = currentDet();
    if (!det) return;
    if (e.key === "Escape" && activeKp) {
        setActiveKp(null);
        return;
    }
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= kpSchema.length) {
        const name = kpSchema[n - 1].name;
        if (applicableKps(det).some(k => k.name === name)) setActiveKp(name);
    }
}

function drawKeypoints() {
    const det = currentDet();
    if (!det || !det.keypoints) return;
    const kp = det.keypoints;
    const P = (name) => kp[name] ? [kp[name][0] * canvasScale, kp[name][1] * canvasScale] : null;

    const line = (a, b, color, dash = []) => {
        if (!a || !b) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash(dash);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.setLineDash([]);
    };

    // M3 ekseni, M2 ekseni (apeks → oklüzal orta), M2 oklüzal hattı
    line(P("m3_apex"), P("m3_crown"), THEME.axis);
    const cm = P("m2_cusp_mesial"), cd = P("m2_cusp_distal");
    if (cm && cd) {
        line(cm, cd, THEME.occlusal, [5, 4]);
        line(P("m2_apex"), [(cm[0] + cd[0]) / 2, (cm[1] + cd[1]) / 2], "#c9a7ff", [3, 3]);
    }

    kpSchema.forEach((k, i) => {
        const p = P(k.name);
        if (!p) return;
        ctx.fillStyle = KP_COLORS[i];
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = `500 11px ${THEME.mono}`;
        ctx.fillStyle = "#fff";
        ctx.fillText(String(i + 1), p[0] + 7, p[1] - 6);
    });
}

function requestGeometry() {
    clearTimeout(geomTimer);
    geomTimer = setTimeout(async () => {
        const det = currentDet();
        if (!det) return;
        const reqIdx = det.index;
        try {
            const res = await fetch("/api/label/geometry", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keypoints: det.keypoints, jaw: jawOf(det), bbox: det.bbox }),
            });
            if (selectedIdx !== reqIdx) return;
            lastGeometry = await res.json();
            renderSuggestions();
        } catch (e) {
            $("kpSuggest").innerHTML = "";
        }
    }, 150);
}

const SUGGESTION_FIELDS = [
    ["impaction", "Açı", "impactionGroup", g => g.angle !== undefined ? `${g.angle}° · ${g.reference}` : ""],
    ["depth", "Derinlik", "depthGroup", () => ""],
    ["ramus", "Ramus", "ramusGroup", g => g.space_ratio !== undefined ? `boşluk/genişlik ${g.space_ratio}` : ""],
    ["nerve", "Sinir", "nerveGroup", g => g.gap_ratio !== undefined ? `mesafe/boy ${g.gap_ratio}` : ""],
];

function renderSuggestions() {
    const box = $("kpSuggest");
    box.innerHTML = "";
    const g = lastGeometry;
    const det = currentDet();
    if (!g || !det || !Object.keys(det.keypoints).length) return;

    SUGGESTION_FIELDS.forEach(([key, name, group, metric]) => {
        const r = g[key];
        if (!r) return;
        const applied = getRadio(group) === r.value;
        const row = document.createElement("div");
        row.className = "sg-row" + (applied ? " applied" : "");
        const left = document.createElement("div");
        left.innerHTML = `<span class="sg-name">${name}</span> `;
        const val = document.createElement("span");
        val.className = "sg-val";
        val.textContent = r.value;
        const m = document.createElement("span");
        m.className = "sg-metric";
        m.textContent = metric(r);
        left.append(val, m);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = applied ? "seçili" : "uygula";
        btn.addEventListener("click", () => { setRadio(group, r.value); renderSuggestions(); });
        row.append(left, btn);
        box.appendChild(row);
    });

    if (g.missing && g.missing.length) {
        const note = document.createElement("div");
        note.className = "sg-note";
        const names = g.missing.map(n => (kpSchema.find(k => k.name === n) || {}).label || n);
        note.textContent = `Eksik nokta: ${names.join(", ")}`;
        box.appendChild(note);
    }
}
