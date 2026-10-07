# Yirmilik — Gömülü 3. Molar Karar Destek Sistemi

Panoramik röntgenden yirmilik dişleri (3. molarları) tespit eden, her diş için
Winter açısı ve Pell & Gregory sınıflarını öneren ve **Pederson zorluk indeksi**
ile cerrahi zorluğu değerlendiren klinik karar destek aracı.

> Karar destek aracıdır; tanı koymaz. Klinik karar hekime aittir.

## Özellikler

| Sayfa | İçerik |
|---|---|
| `/` Analiz | Röntgen görüntüleyici (zoom/pan, parlaklık, kontrast, negatif), YOLO ile yirmilik tespiti, FDI çene haritası (18/28/38/48), diş ekseni + oklüzal düzlem katmanları, bulgu bazında güven göstergesi, otomatik Pederson değerlendirmesi |
| `/label` Etiketleme | Hekim etiketleme aracı: kutu çizme, Winter / P&G / kök / sinir etiketleri, **8 anatomik nokta** işaretleme ve noktalardan geometrik sınıf önerisi |
| `/admin` Yönetim | Toplu görüntü yükleme, etiketli veri setini ZIP olarak indirme (YOLO + YOLO-pose + klinik JSON) |

## Mimari

```
app.py             FastAPI — sayfalar ve API
config.py          Model yolu, eşikler, seçenek listeleri
image_analyzer.py  Sezgisel analiz: çene (üst/alt), PCA diş ekseni + mine ile kron yönü,
                   oklüzal düzlem, ramus kenarı, FDI numarası
geometry.py        Anatomik noktalardan Winter açısı (M2 eksenine göre), P&G derinlik/ramus,
                   sinir yakınlığı — etiketleme aracı ve ileride pose modeli kullanır
scorer.py          Pederson indeksi (3-10) + risk faktörleriyle kategori yükseltme
post_filter.py     Anatomik yanlış-pozitif filtresi
label_storage.py   Klinik etiketlerin JSON depolaması ve dışa aktarımı
static/            theme.css (ortak tema), index/app.*, label.*, admin.html
```

### Değerlendirme mantığı

- **Pederson indeksi** = açı (Mesio 1, Yatay 2, Dikey 3, Disto/Ters 4) + derinlik (A 1, B 2, C 3) + ramus (1/2/3).
  3-4 az zor · 5-6 orta zor · 7-10 çok zor.
- **Risk faktörleri** puana eklenmez (literatürde doğrulanmış ağırlık yok); açık bir kuralla kategoriyi yükseltir:
  1 büyük risk (sinir teması, ağız açıklığı < 30 mm) **veya** 2+ orta risk (yaş > 30, kısıtlı açıklık, eğri/ayrık kök) → bir seviye yukarı.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                     # http://localhost:7860
```

### Ortam değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MODEL_PATH` | `trained_models/disprojesi52/weights/best.pt` | Tespit modeli |
| `INFERENCE_DEVICE` | `cpu` | Çıkarım cihazı (`0` = ilk GPU) |
| `APP_USER`, `APP_PASSWORD` | — | Tanımlıysa tüm site HTTP Basic Auth ile korunur |
| `ADMIN_TOKEN` | — | Tanımlı değilse yükleme ve dışa aktarma **kapalıdır** |
| `IMAGES_DIR` | `train/images` | Etiketleme görüntüleri (Railway: `/data/images`) |
| `LABELS_DIR` | `labels_clinical` | Klinik etiketler (Railway: `/data/labels_clinical`) |
| `MAX_UPLOAD_MB` | `20` | Yükleme boyut sınırı |

> Canlı ortamda `APP_USER`/`APP_PASSWORD` ve `ADMIN_TOKEN` mutlaka tanımlanmalıdır: röntgenler hasta verisidir.

## Veri ve eğitim

### 1. Sızıntısız veri bölmesi

Roboflow dışa aktarımı aynı röntgenin kopyalarını (`*.rf.<hash>`) ve aynı hastanın farklı
görüntülerini farklı bölmelere dağıtıyordu; eski metrikler bu yüzden şişikti.

```bash
python veri_bol.py      # → dataset_v2/ + data_v2.yaml (70/15/15, grup bazlı)
```

Gruplama: aynı kaynak dosya, hasta öneki (`04269002-I4` → `04269002`) ve görsel kopya
(küçültülmüş görüntü korelasyonu ≥ 0.945, elle doğrulanmış eşik).

### 2. Tespit modeli

```bash
python train.py         # YOLO11s, data_v2.yaml; bitince test setinde ölçüm yazdırır
```

### 3. Anatomik nokta (pose) modeli

1. Hekimler `/label` sayfasında dişlere 8 nokta işaretler.
2. `/admin` → **Veri setini indir (ZIP)**.
3. Eğitim:

```bash
python train_pose.py wisdom_teeth_yolo_dataset.zip --dry-run   # veri kontrolü
python train_pose.py wisdom_teeth_yolo_dataset.zip
```

Noktalar: M3 kron tepesi, M3 apeksi, M2 mesial/distal tüberkül, M2 distal CEJ, M2 apeksi,
ramus ön kenarı, mandibular kanal üst sınırı (son ikisi yalnız alt çene).

## Test

```bash
pytest
```

Geometri, skor, sezgisel analiz (sentetik görüntüler), API (geçici klasörlerde) ve pose veri hazırlığı testleri.

## Bilinen sınırlar

- Sezgisel analizin (açı/derinlik/ramus) hekim etiketlerine karşı doğruluğu henüz **ölçülmedi**;
  arayüzdeki güven değerleri yaklaşık değerlerdir. Ölçüm için etiketleme aracıyla klinik etiket toplanmalıdır.
- Sezgisel Winter açısı 2. molar yerine görüntü dikeyine göre ölçülür; noktalı etiketleme ve pose modeli bunu giderir.
- Kırpılmış röntgenlerde, karşı çene görünmüyorsa çene tespiti sabit eşiğe düşer (~%91 doğruluk).

## Lisans

Veri seti: Roboflow Universe, CC BY 4.0. Kod: MIT.
