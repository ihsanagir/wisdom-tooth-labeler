/**
 * Yirmilik — Karar Destek ön yüzü
 * Röntgen görüntüleyici (zoom/pan/filtre), SVG anotasyon katmanı,
 * FDI çene haritası ve diş bazlı Pederson değerlendirmesi.
 */

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const LOW_DET_CONF = 0.55;     // altı: kesikli kutu
const REVIEW_CONF = 0.5;       // altı: "hekim kontrolü" uyarısı

const state = {
    options: null,
    patient: { gender: null, age: null, mouth_opening: null },
    file: null,
    imgURL: null,
    nat: { w: 0, h: 0 },
    view: { s: 1, x: 0, y: 0, fitS: 1 },
    adjust: { b: 100, c: 100, inv: false },
    layers: { boxes: true, axis: true, occlusal: true },
    layersHidden: false,
    detections: [],
    teeth: [],          // detections ile aynı sırada: {impaction, depth, ramus, root, nerve, edited:{}, result}
    selected: -1,
};

const el = {
    stage: $('stage'), canvas: $('canvas'), img: $('xray'), overlay: $('overlay'),
    dropzone: $('dropzone'), fileInput: $('fileInput'),
    btnAnalyze: $('btnAnalyze'), analyzeLabel: $('analyzeLabel'),
    zoomReadout: $('zoomReadout'), stageStatus: $('stageStatus'), btnClear: $('btnClear'),
    toolbar: document.querySelector('.toolbar'),
};

// ════════════════════════════════════════════════════════════════════════
// Başlangıç
// ════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    updateToolbarState();
    renderArch();
    bindViewer();
    bindKeyboard();
    checkHealth();

    try {
        state.options = await fetch('/api/options').then(r => r.json());
    } catch {
        showStatus('Sunucuya bağlanılamadı.', true);
        return;
    }
    const o = state.options;
    state.patient = { gender: o.gender[0], age: o.age[0], mouth_opening: o.mouth_opening[0] };
    segmented('seg-gender', o.gender, state.patient.gender, v => setPatient('gender', v));
    segmented('seg-age', o.age, state.patient.age, v => setPatient('age', v),
        v => v === '>30' ? 'risk-moderate' : '');
    segmented('seg-mouth', o.mouth_opening, state.patient.mouth_opening, v => setPatient('mouth_opening', v),
        v => v.startsWith('Çok') ? 'risk-major' : v.startsWith('Kısıtlı') ? 'risk-moderate' : '');
});

async function checkHealth() {
    const pill = $('statusPill');
    try {
        const h = await fetch('/health').then(r => r.json());
        pill.classList.add(h.model_loaded ? 'ok' : 'bad');
        $('statusText').textContent = h.model_loaded ? `Model hazır · v${h.version}` : 'Model yüklenemedi';
    } catch {
        pill.classList.add('bad');
        $('statusText').textContent = 'Sunucu yanıt vermiyor';
    }
}

function setPatient(key, value) {
    state.patient[key] = value;
    state.teeth.forEach((_, i) => evaluate(i));
}

// ════════════════════════════════════════════════════════════════════════
// Segment kontrolü (erişilebilir radio grubu)
// ════════════════════════════════════════════════════════════════════════
function segmented(containerId, options, value, onChange, riskClass = () => '') {
    const box = $(containerId);
    box.innerHTML = '';
    options.forEach(opt => {
        const b = document.createElement('button');
        b.type = 'button';
        b.setAttribute('role', 'radio');
        b.setAttribute('aria-checked', String(opt === value));
        b.tabIndex = opt === value ? 0 : -1;
        b.textContent = opt;
        const rc = riskClass(opt);
        if (rc) b.classList.add(rc);
        b.addEventListener('click', () => select(b, opt));
        box.appendChild(b);
    });
    box.onkeydown = (e) => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
        e.preventDefault();
        const btns = [...box.children];
        const cur = btns.findIndex(b => b.getAttribute('aria-checked') === 'true');
        const step = (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 1;
        const next = btns[(cur + step + btns.length) % btns.length];
        next.click();
        next.focus();
    };
    function select(btn, opt) {
        [...box.children].forEach(b => {
            b.setAttribute('aria-checked', String(b === btn));
            b.tabIndex = b === btn ? 0 : -1;
        });
        onChange(opt);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Görüntü yükleme
// ════════════════════════════════════════════════════════════════════════
function loadFile(file) {
    if (!file) return;
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        showStatus('Lütfen JPG veya PNG formatında bir röntgen seçin.', true);
        return;
    }
    clearImage();
    state.file = file;
    state.imgURL = URL.createObjectURL(file);
    el.img.onload = () => {
        state.nat = { w: el.img.naturalWidth, h: el.img.naturalHeight };
        el.img.width = state.nat.w;
        el.img.height = state.nat.h;
        el.overlay.setAttribute('viewBox', `0 0 ${state.nat.w} ${state.nat.h}`);
        el.overlay.setAttribute('width', state.nat.w);
        el.overlay.setAttribute('height', state.nat.h);
        el.dropzone.hidden = true;
        el.canvas.hidden = false;
        el.btnClear.hidden = false;
        el.stage.classList.remove('empty');
        el.btnAnalyze.disabled = false;
        fitView();
        updateToolbarState();
    };
    el.img.src = state.imgURL;
}

function clearImage() {
    if (state.imgURL) URL.revokeObjectURL(state.imgURL);
    Object.assign(state, { file: null, imgURL: null, detections: [], teeth: [], selected: -1 });
    el.img.removeAttribute('src');
    el.overlay.innerHTML = '';
    el.canvas.hidden = true;
    el.dropzone.hidden = false;
    el.btnClear.hidden = true;
    el.stage.classList.add('empty');
    el.btnAnalyze.disabled = true;
    el.fileInput.value = '';
    hideStatus();
    $('teethPanel').hidden = true;
    $('teethEmpty').hidden = false;
    $('summaryBlock').hidden = true;
    renderArch();
    updateToolbarState();
}

function updateToolbarState() {
    el.toolbar.classList.toggle('no-image', !state.imgURL);
    el.toolbar.classList.toggle('no-result', !state.detections.length);
    document.querySelectorAll('.stage-hint').forEach(h => { h.hidden = !state.imgURL; });
}

// ════════════════════════════════════════════════════════════════════════
// Görüntüleyici: zoom / pan / filtre
// ════════════════════════════════════════════════════════════════════════
function bindViewer() {
    $('btnUpload').addEventListener('click', () => el.fileInput.click());
    $('btnBrowse').addEventListener('click', () => el.fileInput.click());
    el.fileInput.addEventListener('change', e => loadFile(e.target.files[0]));
    el.btnClear.addEventListener('click', clearImage);
    el.btnAnalyze.addEventListener('click', runDetection);

    $('btnZoomIn').addEventListener('click', () => zoomAtCenter(1.25));
    $('btnZoomOut').addEventListener('click', () => zoomAtCenter(0.8));
    $('btnFit').addEventListener('click', fitView);

    $('rngBrightness').addEventListener('input', e => { state.adjust.b = +e.target.value; applyFilter(); });
    $('rngContrast').addEventListener('input', e => { state.adjust.c = +e.target.value; applyFilter(); });
    $('btnInvert').addEventListener('click', toggleInvert);

    document.querySelectorAll('[data-layer]').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.layer;
            state.layers[key] = !state.layers[key];
            btn.setAttribute('aria-pressed', String(state.layers[key]));
            applyLayers();
        });
    });

    // Sürükle-bırak
    ['dragenter', 'dragover'].forEach(t => el.stage.addEventListener(t, e => {
        e.preventDefault();
        el.dropzone.classList.add('over');
    }));
    ['dragleave', 'drop'].forEach(t => el.stage.addEventListener(t, e => {
        e.preventDefault();
        el.dropzone.classList.remove('over');
    }));
    el.stage.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));

    // Tekerlek ile imleç etrafında zoom
    el.stage.addEventListener('wheel', e => {
        if (!state.imgURL) return;
        e.preventDefault();
        const r = el.stage.getBoundingClientRect();
        zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    // Sürükleyerek kaydırma; az hareketli tıklama = kutu seçimi
    let drag = null;
    el.stage.addEventListener('pointerdown', e => {
        if (!state.imgURL || e.button !== 0 || e.target.closest('button')) return;
        drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y, moved: false,
                 tooth: e.target.closest('.tooth') };
        el.stage.setPointerCapture(e.pointerId);
    });
    el.stage.addEventListener('pointermove', e => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (!drag.moved && Math.hypot(dx, dy) < 4) return;
        drag.moved = true;
        el.stage.classList.add('dragging');
        state.view.x = drag.vx + dx;
        state.view.y = drag.vy + dy;
        applyView();
    });
    const endDrag = () => {
        if (drag && !drag.moved && drag.tooth) selectTooth(+drag.tooth.dataset.i);
        drag = null;
        el.stage.classList.remove('dragging');
    };
    el.stage.addEventListener('pointerup', endDrag);
    el.stage.addEventListener('pointercancel', endDrag);

    window.addEventListener('resize', () => { if (state.imgURL) fitView(); });
}

function fitView() {
    const r = el.stage.getBoundingClientRect();
    const s = Math.min(r.width / state.nat.w, r.height / state.nat.h) * 0.96;
    state.view = { s, fitS: s, x: (r.width - state.nat.w * s) / 2, y: (r.height - state.nat.h * s) / 2 };
    applyView();
}

function zoomAt(factor, cx, cy) {
    const v = state.view;
    const ns = Math.min(Math.max(v.s * factor, v.fitS * 0.5), v.fitS * 12);
    v.x = cx - (cx - v.x) * (ns / v.s);
    v.y = cy - (cy - v.y) * (ns / v.s);
    v.s = ns;
    applyView();
}

function zoomAtCenter(factor) {
    const r = el.stage.getBoundingClientRect();
    zoomAt(factor, r.width / 2, r.height / 2);
}

function applyView() {
    const v = state.view;
    el.canvas.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
    el.zoomReadout.textContent = `${Math.round((v.s / v.fitS) * 100)}%`;
}

function applyFilter() {
    const a = state.adjust;
    el.img.style.filter = `brightness(${a.b}%) contrast(${a.c}%)${a.inv ? ' invert(1)' : ''}`;
}

function toggleInvert() {
    state.adjust.inv = !state.adjust.inv;
    $('btnInvert').setAttribute('aria-pressed', String(state.adjust.inv));
    applyFilter();
}

function applyLayers() {
    const o = el.overlay.classList;
    o.toggle('hide-boxes', !state.layers.boxes);
    o.toggle('hide-axis', !state.layers.axis);
    o.toggle('hide-occlusal', !state.layers.occlusal);
    o.toggle('hide-all', state.layersHidden);
}

// ════════════════════════════════════════════════════════════════════════
// Tespit
// ════════════════════════════════════════════════════════════════════════
async function runDetection() {
    if (!state.file) return;
    el.btnAnalyze.disabled = true;
    el.btnAnalyze.classList.add('loading');
    el.analyzeLabel.textContent = 'Analiz ediliyor…';
    showStatus('Röntgen analiz ediliyor…');

    try {
        const fd = new FormData();
        fd.append('file', state.file);
        const res = await fetch('/api/detect', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Tespit başarısız.');

        state.detections = data.detections || [];
        state.teeth = state.detections.map(d => ({
            impaction: d.auto_analysis.impaction,
            depth: d.auto_analysis.depth,
            ramus: d.auto_analysis.ramus,
            root: state.options.root[0],
            nerve: state.options.nerve[0],
            edited: {},
            result: null,
        }));

        $('statCount').textContent = data.count;
        $('statFiltered').textContent = data.filtered_count;
        $('summaryBlock').hidden = false;

        if (!state.detections.length) {
            showStatus('Yirmilik diş tespit edilemedi.', true);
            $('teethPanel').hidden = true;
            $('teethEmpty').hidden = false;
        } else {
            hideStatus();
            $('teethEmpty').hidden = true;
            $('teethPanel').hidden = false;
            renderTabs();
            selectTooth(0);
            await Promise.all(state.teeth.map((_, i) => evaluate(i)));
        }
        renderOverlay();
        renderArch();
        updateToolbarState();
    } catch (err) {
        showStatus(err.message, true);
    } finally {
        el.btnAnalyze.disabled = !state.file;
        el.btnAnalyze.classList.remove('loading');
        el.analyzeLabel.textContent = 'Yeniden analiz et';
    }
}

// ════════════════════════════════════════════════════════════════════════
// SVG anotasyon katmanı (görüntü koordinatlarında, zoomla birlikte ölçeklenir)
// ════════════════════════════════════════════════════════════════════════
function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (parent) parent.appendChild(node);
    return node;
}

function renderOverlay() {
    el.overlay.innerHTML = '';
    const fs = Math.max(14, state.nat.w * 0.012);

    state.detections.forEach((d, i) => {
        const a = d.auto_analysis;
        const [x1, y1, x2, y2] = d.bbox;
        const w = x2 - x1;
        const g = svg('g', { class: 'tooth', 'data-i': i }, el.overlay);

        svg('line', { class: 'occl', x1: x1 - w * 0.5, x2: x2 + w * 0.5, y1: a.occlusal_y, y2: a.occlusal_y }, g);
        svg('rect', {
            class: 'box' + (d.confidence < LOW_DET_CONF ? ' low' : ''),
            x: x1, y: y1, width: w, height: y2 - y1, rx: fs * 0.25,
        }, g);
        if (a.tooth_axis) {
            const [[rx, ry], [cx, cy]] = a.tooth_axis;
            svg('line', { class: 'axis', x1: rx, y1: ry, x2: cx, y2: cy }, g);
            svg('circle', { class: 'crown', cx, cy, r: fs * 0.32 }, g);
        }

        const label = toothLabel(i);
        const tag = svg('g', { class: 'tag' }, g);
        const tw = fs * (0.62 * label.length + 0.9);
        svg('rect', { x: x1, y: y1 - fs * 1.5, width: tw, height: fs * 1.35, rx: fs * 0.25 }, tag);
        const t = svg('text', { x: x1 + fs * 0.45, y: y1 - fs * 0.48, 'font-size': fs }, tag);
        t.textContent = label;
    });

    applyLayers();
    highlightSelection();
}

function highlightSelection() {
    el.overlay.querySelectorAll('.tooth').forEach(g => {
        const i = +g.dataset.i;
        g.classList.toggle('selected', i === state.selected);
        g.classList.toggle('dim', state.selected >= 0 && i !== state.selected);
    });
}

function toothLabel(i) {
    const fdi = state.detections[i].auto_analysis.fdi;
    const dup = state.detections.filter(d => d.auto_analysis.fdi === fdi).length > 1;
    return dup ? `${fdi}·${i + 1}` : String(fdi);
}

// ════════════════════════════════════════════════════════════════════════
// Çene haritası (FDI)
// ════════════════════════════════════════════════════════════════════════
function renderArch() {
    const map = $('archMap');
    map.innerHTML = '';
    // Panoramikte oklüzal düzlem "gülümseme" eğrisidir: ortada aşağıda, uçlarda yukarıda
    const cx = 130, pitch = 14.5, tw = 12, th = 16;
    const smile = dx => 22 * (1 - (dx / 116) ** 2);
    const upperY = 34, lowerY = 54;   // alt sıra = üst sıra + diş boyu + aralık
    const upper = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28];
    const lower = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];

    const occlY = upperY + th + 2;
    svg('path', { class: 'gum', d: `M6 ${occlY + smile(124)} Q130 ${occlY + 2 * smile(0)} 254 ${occlY + smile(124)}` }, map);
    const side = (x, txt, anchor) => {
        const t = svg('text', { class: 'side', x, y: 10, 'text-anchor': anchor }, map);
        t.textContent = txt;
    };
    side(4, 'HASTA SAĞI', 'start');
    side(256, 'HASTA SOLU', 'end');

    const found = {};
    state.detections.forEach((d, i) => {
        const f = d.auto_analysis.fdi;
        if (!(f in found)) found[f] = i;
    });

    [[upper, upperY, -1], [lower, lowerY, 1]].forEach(([row, baseY, dir]) => {
        row.forEach((num, k) => {
            const dx = (k - 7.5) * pitch;
            const x = cx + dx - tw / 2;
            const y = baseY + smile(dx);
            const isWisdom = num % 10 === 8;
            if (!isWisdom) {
                svg('rect', { class: 't', x, y, width: tw, height: th, rx: 3.5 }, map);
                return;
            }
            const i = found[num];
            const sev = i !== undefined ? state.teeth[i]?.result?.severity || '' : '';
            const cls = ['w', i !== undefined ? 'found' : '', sev, i === state.selected && i !== undefined ? 'selected' : '']
                .filter(Boolean).join(' ');
            const node = svg('rect', { class: cls, x: x - 1, y: y - 1, width: tw + 2, height: th + 2, rx: 4 }, map);
            if (i !== undefined) node.addEventListener('click', () => selectTooth(i));
            const ny = dir < 0 ? y - 7 : y + th + 15;
            const lbl = svg('text', { class: 'num' + (i !== undefined ? ' found' : ''), x: x + tw / 2, y: ny }, map);
            lbl.textContent = num;
        });
    });
}

// ════════════════════════════════════════════════════════════════════════
// Diş sekmeleri ve detay paneli
// ════════════════════════════════════════════════════════════════════════
function renderTabs() {
    const tabs = $('toothTabs');
    tabs.innerHTML = '';
    state.detections.forEach((d, i) => {
        const b = document.createElement('button');
        b.className = 'tooth-tab';
        b.setAttribute('role', 'tab');
        b.setAttribute('aria-selected', String(i === state.selected));
        const sev = state.teeth[i].result?.severity || '';
        b.innerHTML = `<span class="sev ${sev}"></span>`;
        b.append(toothLabel(i));
        b.addEventListener('click', () => selectTooth(i));
        tabs.appendChild(b);
    });
}

function selectTooth(i) {
    if (i < 0 || i >= state.detections.length) return;
    state.selected = i;
    renderTabs();
    renderDetail();
    highlightSelection();
    renderArch();
}

function renderDetail() {
    const i = state.selected;
    const d = state.detections[i];
    const a = d.auto_analysis;
    const t = state.teeth[i];
    const o = state.options;
    const patientSide = (d.bbox[0] + d.bbox[2]) / 2 < state.nat.w / 2 ? 'hasta sağı' : 'hasta solu';

    $('dFdi').textContent = a.fdi;
    $('dTitle').textContent = `${a.jaw} · ${patientSide}`;
    $('dSub').textContent = `Tespit güveni %${Math.round(d.confidence * 100)} · eksen ${a.angle_value}°`;

    bindFinding('impaction', o.impaction, a.impaction_confidence);
    bindFinding('depth', o.depth, a.depth_confidence);
    bindFinding('ramus', o.ramus, a.ramus_confidence);

    segmented('seg-root', o.root, t.root, v => updateTooth(i, 'root', v),
        v => v === o.root[0] ? '' : 'risk-moderate');
    segmented('seg-nerve', o.nerve, t.nerve, v => updateTooth(i, 'nerve', v),
        v => v === o.nerve[0] ? '' : 'risk-major');

    renderResult();
}

function bindFinding(key, options, conf) {
    const i = state.selected;
    const t = state.teeth[i];
    const sel = $(`sel-${key}`);
    sel.innerHTML = '';
    options.forEach(opt => sel.add(new Option(opt, opt, false, opt === t[key])));
    sel.onchange = () => {
        t.edited[key] = true;
        updateTooth(i, key, sel.value);
        bindFinding(key, options, conf);
    };

    const box = $(`f-${key}`);
    const edited = !!t.edited[key];
    const review = !edited && conf < REVIEW_CONF;
    sel.classList.toggle('edited', edited);
    box.classList.toggle('review', review);
    box.classList.toggle('good', !edited && !review);
    box.querySelector('.conf-bar i').style.width = edited ? '100%' : `${Math.round(conf * 100)}%`;
    box.querySelector('.conf-text').textContent = edited
        ? 'Hekim tarafından düzeltildi'
        : review ? `Düşük güven %${Math.round(conf * 100)} — kontrol edin`
                 : `Güven %${Math.round(conf * 100)}`;
}

function updateTooth(i, key, value) {
    state.teeth[i][key] = value;
    evaluate(i);
}

// ════════════════════════════════════════════════════════════════════════
// Pederson değerlendirmesi
// ════════════════════════════════════════════════════════════════════════
async function evaluate(i) {
    const t = state.teeth[i];
    if (!t) return;
    const body = {
        ...state.patient,
        impaction: t.impaction, depth: t.depth, ramus: t.ramus, root: t.root, nerve: t.nerve,
    };
    try {
        const res = await fetch('/api/analyze', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        t.result = res.ok ? data : { error: data.error || 'Değerlendirme başarısız.' };
    } catch (err) {
        t.result = { error: err.message };
    }
    if (state.teeth[i] !== t) return;  // bu arada yeni görüntü yüklendiyse
    if (i === state.selected) renderResult();
    renderTabs();
    renderArch();
}

function renderResult() {
    const box = $('result');
    const r = state.teeth[state.selected]?.result;
    box.innerHTML = '';
    if (!r) return;
    if (r.error) {
        box.innerHTML = `<div class="result-card"><span class="risk major">${escapeHtml(r.error)}</span></div>`;
        return;
    }

    const sev = r.severity;
    const zone = n => (n <= 4 ? 'simple' : n <= 6 ? 'surgical' : 'advanced');
    const cells = [];
    const labels = [];
    for (let n = 3; n <= 10; n++) {
        cells.push(`<span class="z-${zone(n)}${n <= r.pederson_index ? ' on' : ''}"></span>`);
        labels.push(`<span>${n}</span>`);
    }
    const rows = r.breakdown.map(b =>
        `<tr><td>${escapeHtml(b.factor)}</td><td>${escapeHtml(b.value)}</td><td class="pts">+${b.points}</td></tr>`).join('');
    const risks = r.risk_factors.map(f =>
        `<span class="risk ${f.level}">${escapeHtml(f.factor)}: ${escapeHtml(f.value)}</span>`).join('');
    const notes = r.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('');

    // Öneri metninin ilk bloğu: başlık + açıklama (klinik notlar risk etiketlerinde zaten var)
    const [head, ...rest] = r.recommendation.split('\n\n')[0].split('\n');
    const headText = head.replace(/^[^\p{L}]+/u, '');

    box.innerHTML = `
        <div class="result-card">
            <div class="result-top">
                <div>
                    <div class="result-index ${sev}">${r.pederson_index}<small>/${r.pederson_max}</small></div>
                    <div class="result-caption">Pederson zorluk indeksi</div>
                </div>
                <span class="sev-chip ${sev}">${escapeHtml(r.severity_label)}</span>
            </div>
            <div class="scale" aria-hidden="true">${cells.join('')}</div>
            <div class="scale-labels" aria-hidden="true">${labels.join('')}</div>
            <table class="breakdown">${rows}</table>
            ${risks ? `<div class="risks">${risks}</div>` : ''}
            ${notes ? `<ul class="notes">${notes}</ul>` : ''}
            <div class="recommendation"><b>${escapeHtml(headText)}</b>\n${escapeHtml(rest.join('\n'))}</div>
        </div>`;
}

// ════════════════════════════════════════════════════════════════════════
// Yardımcılar
// ════════════════════════════════════════════════════════════════════════
function bindKeyboard() {
    document.addEventListener('keydown', e => {
        if (e.target.closest?.('input, select, textarea')) return;
        if (!state.imgURL) return;
        switch (e.key) {
            case ' ':
                if (e.target.closest?.('button')) return;
                e.preventDefault();
                state.layersHidden = !state.layersHidden;
                applyLayers();
                break;
            case 'i': case 'I': toggleInvert(); break;
            case '+': case '=': zoomAtCenter(1.25); break;
            case '-': zoomAtCenter(0.8); break;
            case '0': fitView(); break;
            case 'Escape': clearImage(); break;
        }
    });
}

function showStatus(msg, isError = false) {
    el.stageStatus.textContent = msg;
    el.stageStatus.classList.toggle('error', isError);
    el.stageStatus.hidden = false;
}

function hideStatus() {
    el.stageStatus.hidden = true;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}
