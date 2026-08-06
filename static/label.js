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
        imageList.innerHTML = `<div class="list-loading" style="color:#ef4444">Yüklenemedi.</div>`;
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
        div.innerHTML = `
            <span class="img-dot ${img.labeled ? "labeled" : "unlabeled"}"></span>
            <span class="img-name" title="${img.name}">${img.name}</span>
        `;
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
    renderImageList();
    showLabelForm(false);
    toothTabs.innerHTML = `<span style="color:var(--text-muted);font-size:12px">Yükleniyor…</span>`;

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
            });
        });

        // Model tespiti dene (eğer model yüklüyse)
        try {
            const res  = await fetch(`/api/label/detect?image_name=${encodeURIComponent(name)}`);
            const data = await res.json();
            if (!data.error && data.detections) {
                // Model çıktılarını ekle (eğer önceden eklenmediyse)
                data.detections.forEach(d => {
                    if (!detections.some(ex => ex.index === d.index)) {
                        detections.push(d);
                    }
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
            toothTabs.innerHTML = `<span style="color:var(--accent);font-size:12px">💡 Görüntü üzerine fare ile sürükleyerek 20'lik diş kutusu çizebilirsiniz.</span>`;
        }

    } catch (e) {
        toothTabs.innerHTML = `<span style="color:#ef4444;font-size:12px">⚠ ${e.message}</span>`;
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

        let color = "#6b7280";
        if (isSaved)    color = "#22c55e";
        if (isSelected) color = "#fbbf24";

        ctx.strokeStyle = color;
        ctx.lineWidth   = isSelected ? 2.5 : 1.5;
        ctx.strokeRect(sx, sy, sw, sh);

        // Etiket kutusu
        const label = `${det.index}.Diş`;
        ctx.font = "bold 11px Inter, Arial";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(sx, sy - 18, tw + 8, 18);
        ctx.fillStyle = "#000";
        ctx.fillText(label, sx + 4, sy - 5);
    });

    // Halen çizilmekte olan geçici kutu
    if (isDrawing) {
        const x = Math.min(startX, currentX) * canvasScale;
        const y = Math.min(startY, currentY) * canvasScale;
        const w = Math.abs(currentX - startX) * canvasScale;
        const h = Math.abs(currentY - startY) * canvasScale;

        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth   = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        ctx.fillStyle = "#3b82f6";
        ctx.font = "bold 11px Inter, Arial";
        ctx.fillText("Yeni Diş Kutus...", x + 4, y - 5);
    }
}

function initCanvasEvents() {
    canvas.addEventListener("mousedown", (e) => {
        if (!currentImage) return;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) / canvasScale;
        const my = (e.clientY - rect.top)  / canvasScale;

        // Önce var olan bir kutuya mı tıklandı kontrol et
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
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        currentX = (e.clientX - rect.left) / canvasScale;
        currentY = (e.clientY - rect.top)  / canvasScale;
        redrawCanvas();
    });

    canvas.addEventListener("mouseup", () => {
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
        const newIdx = detections.length + 1;
        const newDet = {
            index: newIdx,
            bbox: [x1, y1, x2, y2],
            confidence: 1.0,
            auto_impaction: "Dikey (Vertical)",
            auto_ramus: "Sınıf 1 (Önünde)",
            auto_depth: "Seviye A (Oklüzal)",
        };

        detections.push(newDet);
        renderTabs();
        selectTooth(newIdx);
    });
}

function enableDrawMode() {
    alert("💡 Fare ile görüntü üzerinde sürükleyerek 20'lik diş kutusunu çizebilirsiniz.");
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
    $("formToothTitle").textContent = `${index}. Diş`;

    $("autoHint").innerHTML =
        `📐 <b>Konum:</b> X:[${det.bbox[0]}, ${det.bbox[2]}] Y:[${det.bbox[1]}, ${det.bbox[3]}]`;

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

    if (!confirm(`${selectedIdx}. Diş kutusunu silmek istediğinize emin misiniz?`)) return;

    try {
        await fetch("/api/label/delete_box", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_name: currentImage, bbox_index: selectedIdx })
        });

        delete savedLabels[selectedIdx];

        // Detections dizisinden çıkar ve indeksleri güncelle
        detections = detections.filter(d => d.index !== selectedIdx);
        detections.forEach((d, idx) => { d.index = idx + 1; });

        // SavedLabels anahtarlarını da güncelle
        const newSaved = {};
        Object.keys(savedLabels).forEach((oldIdx, i) => {
            newSaved[i + 1] = savedLabels[oldIdx];
        });
        savedLabels = newSaved;

        renderTabs();
        selectTooth(detections.length > 0 ? 1 : null);
        redrawCanvas();
        updateImageLabeledStatus();
        showFeedback("✓ Kutu silindi.", true);

    } catch (e) {
        showFeedback("⚠ Silme hatası: " + e.message, false);
    }
}

// ============================================================
// SEKME OLUŞTURMA
// ============================================================

function renderTabs() {
    toothTabs.innerHTML = "";
    if (detections.length === 0) {
        toothTabs.innerHTML = `<span style="color:var(--text-muted);font-size:12px">Fare ile sürüklerek 20'lik diş kutusu çizin.</span>`;
        return;
    }
    detections.forEach(det => {
        const btn = document.createElement("button");
        btn.className = "tooth-tab" + (savedLabels[det.index] ? " saved" : "");
        btn.dataset.idx = det.index;
        btn.textContent = `${det.index}. Diş`;
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
        showFeedback("⚠ Lütfen tüm alanları seçin.", false);
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
            showFeedback("✓ Kaydedildi!", true);

            const next = detections.find(d => d.index > selectedIdx && !savedLabels[d.index]);
            if (next) setTimeout(() => selectTooth(next.index), 600);
        } else {
            showFeedback("⚠ Kaydedilemedi.", false);
        }
    } catch (e) {
        showFeedback("⚠ Hata: " + e.message, false);
    } finally {
        $("saveBtnText").textContent = "💾 Etiketleri Kaydet";
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
        $("statsContent").innerHTML = `<span style="color:#ef4444">Yüklenemedi.</span>`;
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
