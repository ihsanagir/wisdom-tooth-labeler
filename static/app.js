/**
 * Akıllı Yirmilik Diş Karar Destek Sistemi — Frontend
 * v3: Canvas Tabanlı Bbox Overlay (görüntüye dokunmaz)
 */

// --- State ---
let uploadedFile = null;
let dropdownOptions = {};

// Overlay state
let originalImageSrc = null;   // Ham röntgen — asla kaybolmaz
let detectionResults = [];     // API'den gelen detections (bbox koordinatları)
let overlayVisible = false;
let analysisReady = false;
let currentOpacity = 0.80;     // Varsayılan kutu opaklığı

// --- DOM Elements ---
const $ = (id) => document.getElementById(id);
const uploadArea       = $('uploadArea');
const fileInput        = $('fileInput');
const imagePreview     = $('imagePreview');
const previewImg       = $('previewImg');
const overlayCanvas    = $('overlayCanvas');
const clearBtn         = $('clearBtn');
const detectBtn        = $('detectBtn');
const detectText       = $('detectText');
const spinner          = $('spinner');
const detectionSummary = $('detectionSummary');
const teethContainer   = $('teethContainer');
const noDetection      = $('noDetection');
const overlayToggleBtn = $('overlayToggleBtn');
const overlayIcon      = $('overlayIcon');
const overlayText      = $('overlayText');
const overlayBadge     = $('overlayBadge');
const overlayBadgeText = $('overlayBadgeText');
const keyboardHint     = $('keyboardHint');

// Canvas context
const ctx = overlayCanvas ? overlayCanvas.getContext('2d') : null;

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
    await loadOptions();
    setupUploadHandlers();
    setupKeyboardShortcuts();

    // Görüntü yüklenince canvas boyutunu güncelle
    previewImg.addEventListener('load', syncCanvasSize);
    window.addEventListener('resize', () => {
        if (overlayVisible) drawBoxes();
    });
});

/**
 * Canvas boyutunu görüntü elementinin gerçek boyutuyla eşitle.
 */
function syncCanvasSize() {
    const rect = previewImg.getBoundingClientRect();
    overlayCanvas.width  = previewImg.naturalWidth;
    overlayCanvas.height = previewImg.naturalHeight;
}

/**
 * API'den dönen detections[] kullanarak canvas üzerine bbox çizer.
 * Görüntüye tek piksel dokunmaz.
 */
function drawBoxes() {
    if (!ctx || !detectionResults.length) return;

    // Canvas piksel boyutunu görüntünün doğal boyutuyla eşitle
    overlayCanvas.width  = previewImg.naturalWidth;
    overlayCanvas.height = previewImg.naturalHeight;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    detectionResults.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const conf = det.confidence;
        const w = x2 - x1;
        const h = y2 - y1;

        // Confidence'a göre renk
        let r, g, b;
        if (conf >= 0.75) { r = 0;   g = 210; b = 110; }  // Yeşil
        else if (conf >= 0.55) { r = 30; g = 170; b = 255; }  // Mavi
        else               { r = 255; g = 100; b = 80;  }  // Kırmızı

        // Bbox arka plan (çok hafif dolgu)
        ctx.globalAlpha = currentOpacity * 0.12;
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x1, y1, w, h);

        // Bbox kenar çizgisi
        ctx.globalAlpha = currentOpacity;
        ctx.strokeStyle = `rgb(${r},${g},${b})`;
        ctx.lineWidth = Math.max(2, overlayCanvas.width * 0.002);
        ctx.strokeRect(x1, y1, w, h);

        // Etiket arka planı (Sadece numara, yüzdelik kaldırıldı, font küçültüldü)
        const label    = `${det.index}. Diş`;
        const fontSize = Math.max(10, overlayCanvas.width * 0.011);
        ctx.font       = `600 ${fontSize}px Inter, sans-serif`;
        const textW    = ctx.measureText(label).width;
        const padX     = fontSize * 0.5;
        const padY     = fontSize * 0.35;
        const tagH     = fontSize + padY * 2;

        ctx.globalAlpha = currentOpacity * 0.88;
        ctx.fillStyle   = `rgb(${r},${g},${b})`;
        ctx.beginPath();
        ctx.roundRect(x1, y1 - tagH, textW + padX * 2, tagH, [4, 4, 0, 0]);
        ctx.fill();

        // Etiket metni
        ctx.globalAlpha = currentOpacity;
        ctx.fillStyle   = '#ffffff';
        ctx.fillText(label, x1 + padX, y1 - padY);
    });

    // Sonraki çizimler için alpha'yı sıfırla
    ctx.globalAlpha = 1;
}

/**
 * Canvas'ı temizler ve gizler (overlay kapalıyken).
 */
function clearCanvas() {
    if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCanvas.style.display = 'none';
}

// --- API Options ---
async function loadOptions() {
    try {
        const res = await fetch('/api/options');
        dropdownOptions = await res.json();
        fillSelect('gender', dropdownOptions.gender);
        fillSelect('age', dropdownOptions.age);
        fillSelect('mouthOpening', dropdownOptions.mouth_opening);
    } catch (err) {
        console.error('Seçenekler yüklenemedi:', err);
    }
}

function fillSelect(id, options) {
    const sel = $(id);
    if (!sel || !options) return;
    sel.innerHTML = '';
    options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        sel.appendChild(o);
    });
}

// --- Upload Handlers ---
function setupUploadHandlers() {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    clearBtn.addEventListener('click', clearImage);
    detectBtn.addEventListener('click', runDetection);
    overlayToggleBtn.addEventListener('click', toggleOverlay);
}

function handleFile(file) {
    const isDicom = file.name.toLowerCase().endsWith('.dcm');
    if (!file.type.startsWith('image/') && !isDicom) {
        alert('Geçerli bir görüntü (.jpg, .png) veya DICOM (.dcm) dosyası seçin.');
        return;
    }

    uploadedFile = file;
    resetOverlayState();

    if (isDicom) {
        originalImageSrc = null;
        previewImg.src = '';
        previewImg.style.display = 'none';
        let ph = imagePreview.querySelector('.dicom-placeholder');
        if (!ph) {
            ph = document.createElement('div');
            ph.className = 'dicom-placeholder';
            imagePreview.insertBefore(ph, overlayCanvas);
        }
        ph.innerHTML = `<span>🩻</span><p>${escapeHtml(file.name)}</p><small>DICOM dosyası hazır</small>`;
        imagePreview.style.display = 'block';
        uploadArea.style.display = 'none';
        detectBtn.disabled = false;
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageSrc = e.target.result;
        previewImg.src = originalImageSrc;
        previewImg.style.display = 'block';
        const ph = imagePreview.querySelector('.dicom-placeholder');
        if (ph) ph.remove();
        imagePreview.style.display = 'block';
        uploadArea.style.display = 'none';
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    uploadedFile = null;
    originalImageSrc = null;
    detectionResults = [];
    analysisReady = false;

    previewImg.src = '';
    previewImg.style.display = 'none';
    clearCanvas();

    const ph = imagePreview.querySelector('.dicom-placeholder');
    if (ph) ph.remove();

    imagePreview.style.display = 'none';
    uploadArea.style.display = 'block';
    detectBtn.disabled = true;
    fileInput.value = '';

    resetOverlayState();

    teethContainer.innerHTML = '';
    noDetection.style.display = 'block';
    teethContainer.appendChild(noDetection);
    detectionSummary.style.display = 'none';
}

function resetOverlayState() {
    overlayVisible = false;
    clearCanvas();
    overlayToggleBtn.style.display = 'none';
    overlayToggleBtn.classList.remove('overlay-active');
    overlayIcon.textContent = '👁';
    overlayText.textContent = 'Tespitleri Göster';
    overlayBadge.classList.remove('visible');
    keyboardHint.style.display = 'none';
    currentOpacity = 0.80;
}

// --- Detection ---
async function runDetection() {
    if (!uploadedFile) return;

    detectBtn.disabled = true;
    detectText.textContent = 'Analiz ediliyor...';
    spinner.style.display = 'inline-block';
    detectionSummary.style.display = 'none';

    // Yeni analiz — eski overlay'i kapat
    if (overlayVisible) {
        overlayVisible = false;
        clearCanvas();
        overlayToggleBtn.classList.remove('overlay-active');
        overlayIcon.textContent = '👁';
        overlayText.textContent = 'Tespitleri Göster';
    }

    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        const res = await fetch('/api/detect', { method: 'POST', body: formData });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Tespit başarısız.');
        }

        const data = await res.json();

        // Bbox verilerini sakla — canvas çizimi için
        detectionResults = data.detections || [];
        analysisReady = detectionResults.length > 0;

        // Özet
        if (data.count > 0) {
            let txt = `<strong>${data.count}</strong> yirmilik diş tespit edildi.`;
            if (data.filtered_count > 0) {
                txt += `<br><span class="filter-info">🔍 ${data.raw_count} ham tespittten ${data.filtered_count} tanesi anatomik filtre ile kaldırıldı.</span>`;
            }
            detectionSummary.innerHTML = txt;
        } else {
            detectionSummary.innerHTML = '⚠️ Tespit edilemedi. Farklı bir görüntü deneyin.';
        }
        detectionSummary.style.display = 'block';

        renderTeethCards(detectionResults);

        if (analysisReady) {
            overlayToggleBtn.style.display = 'flex';
            overlayBadge.classList.add('visible');
            overlayBadgeText.textContent = `${data.count} tespit hazır`;
            keyboardHint.style.display = 'block';
            
            // Analiz biter bitmez otomatik olarak tespitleri göster (Varsayılan Açık)
            if (!overlayVisible) {
                toggleOverlay();
            }
        }

    } catch (err) {
        detectionSummary.innerHTML = `⚠️ Hata: ${escapeHtml(err.message)}`;
        detectionSummary.style.display = 'block';
    } finally {
        detectBtn.disabled = false;
        detectText.textContent = '🔍 20\'lik Dişleri Analiz Et';
        spinner.style.display = 'none';
    }
}

// --- Canvas Overlay Toggle ---
function toggleOverlay() {
    if (!analysisReady || !detectionResults.length) return;

    overlayVisible = !overlayVisible;

    if (overlayVisible) {
        // Canvas'ı göster ve bbox'ları çiz
        overlayCanvas.style.display = 'block';
        drawBoxes();
        overlayToggleBtn.classList.add('overlay-active');
        overlayIcon.textContent = '🔲';
        overlayText.textContent = 'Tespitleri Gizle';
    } else {
        // Canvas'ı temizle ve gizle — görüntüye dokunma
        clearCanvas();
        overlayToggleBtn.classList.remove('overlay-active');
        overlayIcon.textContent = '👁';
        overlayText.textContent = 'Tespitleri Göster';
    }
}



// --- Keyboard Shortcuts ---
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        switch (e.code) {
            case 'Space':
                e.preventDefault();
                if (analysisReady) toggleOverlay();
                break;
            case 'Escape':
                if (uploadedFile) clearImage();
                break;
        }
    });
}

// --- Confidence Helpers ---
function getConfClass(conf) {
    if (conf >= 0.70) return 'conf-high';
    if (conf >= 0.45) return 'conf-medium';
    return 'conf-low';
}

function getConfLabel(conf) {
    const pct = Math.round(conf * 100);
    if (conf >= 0.70) return `✓ %${pct}`;
    if (conf >= 0.45) return `~ %${pct}`;
    return `⚠ %${pct}`;
}

// --- Render Tooth Cards ---
function renderTeethCards(detections) {
    teethContainer.innerHTML = '';
    if (!detections || detections.length === 0) {
        noDetection.style.display = 'block';
        teethContainer.appendChild(noDetection);
        return;
    }
    noDetection.style.display = 'none';
    detections.forEach((det, idx) => teethContainer.appendChild(createToothCard(det, idx)));
    const first = teethContainer.querySelector('.tooth-card');
    if (first) first.classList.add('open');
}

function createToothCard(det, idx) {
    const card = document.createElement('div');
    card.className = 'tooth-card';
    card.id = `tooth-card-${idx}`;
    const auto = det.auto_analysis;
    const impConf = auto.impaction_confidence || 0.5;
    const ramConf = auto.ramus_confidence || 0.5;
    const depConf = auto.depth_confidence || 0.5;
    card.innerHTML = `
        <div class="tooth-card-header" onclick="toggleCard(${idx})">
            <div class="tooth-info">
                <div class="tooth-badge">${det.index}</div>
                <div>
                    <div class="tooth-label">${det.index}. Diş</div>
                    <div class="tooth-confidence">Model Güveni: ${(det.confidence * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="tooth-auto-tags">
                <span class="tag ${getConfClass(impConf)}">${auto.impaction}</span>
                <span class="tag ${getConfClass(depConf)}">${auto.depth}</span>
            </div>
            <span class="tooth-chevron">▼</span>
        </div>
        <div class="tooth-card-body">
            <div class="tooth-card-content">
                <div class="auto-label">🤖 Otomatik Tespit</div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Gömülülük Açısı <span class="conf-indicator ${getConfClass(impConf)}">${getConfLabel(impConf)}</span></label>
                        <select id="impaction-${idx}">${makeOptions(dropdownOptions.impaction, auto.impaction)}</select>
                    </div>
                    <div class="form-group">
                        <label>Ramus İlişkisi <span class="conf-indicator ${getConfClass(ramConf)}">${getConfLabel(ramConf)}</span></label>
                        <select id="ramus-${idx}">${makeOptions(dropdownOptions.ramus, auto.ramus)}</select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Gömülülük Derinliği <span class="conf-indicator ${getConfClass(depConf)}">${getConfLabel(depConf)}</span></label>
                    <select id="depth-${idx}">${makeOptions(dropdownOptions.depth, auto.depth)}</select>
                </div>
                <div class="manual-label">✋ Manuel Giriş</div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Kök Formu</label>
                        <select id="root-${idx}">${makeOptions(dropdownOptions.root)}</select>
                    </div>
                    <div class="form-group">
                        <label>Sinir (IAN) Komşuluğu</label>
                        <select id="nerve-${idx}">${makeOptions(dropdownOptions.nerve)}</select>
                    </div>
                </div>
                <button class="analyze-btn" onclick="analyzeToothBtn(${idx})">⚡ Zorluk Skoru Hesapla</button>
                <div class="score-result" id="score-result-${idx}"></div>
            </div>
        </div>`;
    return card;
}

function makeOptions(options, selectedValue) {
    if (!options) return '';
    return options.map(opt =>
        `<option value="${opt}" ${opt === selectedValue ? 'selected' : ''}>${opt}</option>`
    ).join('');
}

function toggleCard(idx) {
    const card = $(`tooth-card-${idx}`);
    if (card) card.classList.toggle('open');
}

// --- Analyze Tooth ---
async function analyzeToothBtn(idx) {
    const body = {
        gender: $('gender').value,
        age: $('age').value,
        mouth_opening: $('mouthOpening').value,
        impaction: $(`impaction-${idx}`).value,
        ramus: $(`ramus-${idx}`).value,
        depth: $(`depth-${idx}`).value,
        root: $(`root-${idx}`).value,
        nerve: $(`nerve-${idx}`).value,
    };
    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        renderScore(idx, data);
    } catch (err) {
        console.error('Analiz hatası:', err);
    }
}

function renderScore(idx, data) {
    const container = $(`score-result-${idx}`);
    if (!container) return;
    const sev = data.severity;
    const labels = { simple: 'Basit', surgical: 'Cerrahi', advanced: 'İleri Cerrahi' };
    container.innerHTML = `
        <div class="score-header">
            <span class="score-value ${sev}">%${data.score}</span>
            <span class="score-severity ${sev}">${labels[sev] || sev}</span>
        </div>
        <div class="score-bar-container">
            <div class="score-bar ${sev}" style="width: 0%"></div>
        </div>
        <div class="score-recommendation">${escapeHtml(data.recommendation)}</div>`;
    container.style.display = 'block';
    requestAnimationFrame(() => {
        const bar = container.querySelector('.score-bar');
        if (bar) bar.style.width = `${data.score}%`;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
