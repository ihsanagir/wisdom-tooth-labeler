/**
 * Klinik Etiketleme Arayüzü — JavaScript
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

// ============================================================
// DURUM (STATE)
// ============================================================

let allImages       = [];   // {name, labeled}
let currentFilter   = "all";
let currentImage    = null; // seçili görüntü adı
let detections      = [];   // YOLO tespitleri
let savedLabels     = {};   // bbox_index → {impaction, ramus, depth, notes}
let selectedIdx     = null; // aktif diş index'i (1-bazlı)
let imgNaturalW     = 0;
let imgNaturalH     = 0;
let canvasOffsetX   = 0;
let canvasOffsetY   = 0;
let canvasScale     = 1;

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
// GÖRÜNTÜ SEÇİMİ & YOLO TESPİTİ
// ============================================================

async function selectImage(name) {
    currentImage = name;
    selectedIdx  = null;
    detections   = [];
    savedLabels  = {};
    renderImageList();
    showLabelForm(false);
    toothTabs.innerHTML = `<span style="color:var(--text-muted);font-size:12px">Tespit ediliyor…</span>`;

    // Canvas'ı görünür yap, placeholder gizle
    placeholder.style.display = "none";
    canvas.style.display = "block";

    // Görüntüyü çiz
    await drawImageOnCanvas(name);

    // YOLO tespiti
    try {
        const res  = await fetch(`/api/label/detect?image_name=${encodeURIComponent(name)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        detections = data.detections || [];
        imgNaturalW = data.image_width;
        imgNaturalH = data.image_height;

        // Mevcut etiketleri yükle
        const existing = await fetch(`/api/label/existing?image_name=${encodeURIComponent(name)}`);
        const exData   = await existing.json();
        (exData.labels || []).forEach(l => { savedLabels[l.bbox_index] = l; });

        renderTabs();
        redrawCanvas();

        // İlk etiketsiz dişi seç
        const firstUnsaved = detections.find(d => !savedLabels[d.index]);
        selectTooth((firstUnsaved || detections[0])?.index ?? null);

    } catch (e) {
        toothTabs.innerHTML = `<span style="color:#ef4444;font-size:12px">⚠ ${e.message}</span>`;
    }
}

async function drawImageOnCanvas(name) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const wrap  = canvasWrap;
            const maxW  = wrap.clientWidth;
            const maxH  = wrap.clientHeight - 4;
            const scale = Math.min(maxW / img.width, maxH / img.height);

            canvas.width  = img.width  * scale;
            canvas.height = img.height * scale;

            canvasScale   = scale;
            canvasOffsetX = 0;
            canvasOffsetY = 0;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve();
        };
        img.src = `/train-images/${encodeURIComponent(name)}`;
    });
}

function redrawCanvas() {
    if (!currentImage) return;
    drawImageOnCanvas(currentImage).then(() => drawBboxes());
}

// ============================================================
// CANVAS: BBOX ÇİZİMİ
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

        // Renk önceliği: seçili > etiketlendi > bekleyen
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
}

// ============================================================
// CANVAS: TIKLAMA
// ============================================================

canvas.addEventListener("click", (e) => {
    if (detections.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / canvasScale;
    const my = (e.clientY - rect.top)  / canvasScale;

    for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
            selectTooth(det.index);
            return;
        }
    }
});

// ============================================================
// DİŞ SEÇİMİ
// ============================================================

function selectTooth(index) {
    selectedIdx = index;

    // Tab'ları güncelle
    document.querySelectorAll(".tooth-tab").forEach(t => {
        t.classList.toggle("active", parseInt(t.dataset.idx) === index);
    });

    drawBboxes();

    if (index === null) {
        showLabelForm(false);
        return;
    }

    const det = detections.find(d => d.index === index);
    if (!det) return;

    showLabelForm(true);
    $("formToothTitle").textContent = `${index}. Diş`;

    // Otomatik tahmin ipucu
    $("autoHint").innerHTML =
        `🤖 <b>Otomatik tahmin:</b> ${det.auto_impaction} · ${det.auto_ramus} · ${det.auto_depth}`;

    // Kayıtlı etiket varsa doldur, yoksa otomatik tahmini seç
    const saved = savedLabels[index];
    setRadio("impactionGroup", saved?.impaction ?? det.auto_impaction);
    setRadio("ramusGroup",     saved?.ramus     ?? det.auto_ramus);
    setRadio("depthGroup",     saved?.depth     ?? det.auto_depth);
    $("labelNotes").value = saved?.notes ?? "";

    $("saveFeedback").textContent = "";
}

// ============================================================
// SEKME OLUŞTURMA
// ============================================================

function renderTabs() {
    toothTabs.innerHTML = "";
    if (detections.length === 0) {
        toothTabs.innerHTML = `<span style="color:var(--text-muted);font-size:12px">Diş tespit edilemedi.</span>`;
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
}

function buildGroup(containerId, options) {
    const container = $(containerId);
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
    $(id).classList.toggle("open");
}

// ============================================================
// KAYDETME
// ============================================================

async function saveCurrentLabel() {
    if (selectedIdx === null || !currentImage) return;

    const impaction = getRadio("impactionGroup");
    const ramus     = getRadio("ramusGroup");
    const depth     = getRadio("depthGroup");

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
            drawBboxes();
            updateImageLabeledStatus();
            showFeedback("✓ Kaydedildi!", true);

            // Sonraki etiketlenmemiş dişe geç
            const next = detections.find(d => d.index > selectedIdx && !savedLabels[d.index]);
            if (next) setTimeout(() => selectTooth(next.index), 600);
        } else {
            showFeedback("⚠ Kaydedilemedi.", false);
        }
    } catch (e) {
        showFeedback("⚠ Hata: " + e.message, false);
    } finally {
        $("saveBtnText").textContent = "💾 Kaydet";
    }
}

function showFeedback(msg, ok) {
    const el = $("saveFeedback");
    el.textContent = msg;
    el.className   = "save-feedback " + (ok ? "ok" : "err");
    setTimeout(() => { el.textContent = ""; el.className = "save-feedback"; }, 3000);
}

function updateImageLabeledStatus() {
    const allSaved = detections.every(d => savedLabels[d.index]);
    if (!allSaved) return;

    const img = allImages.find(i => i.name === currentImage);
    if (img && !img.labeled) {
        img.labeled = true;
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
    const distHtml = (obj) => Object.entries(obj).length === 0
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
    `;
}
