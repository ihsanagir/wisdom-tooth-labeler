# 🏷️ Wisdom Tooth Labeler

> Custom annotation tool for marking wisdom teeth (3rd molars) on dental panoramic radiographic images, with **YOLO-format export**.

---

## 📌 Overview

A dedicated labeling utility built to streamline the data annotation pipeline for the [20lik_dis_algilama](https://github.com/ihsanagir/20lik_dis_algilama) YOLO detection project.

**Supports:**
- Bounding box drawing on dental radiograph images
- YOLO format export (`.txt` label files)
- Class labeling for different wisdom tooth positions

---

## 🛠️ Stack

```
Language     →  Python
Interface    →  (Tkinter / Streamlit / Custom)
Output       →  YOLO format (.txt bounding boxes)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ihsanagir/wisdom-tooth-labeler.git
cd wisdom-tooth-labeler
pip install -r requirements.txt
python app.py
```

---

## 🗂️ Output Format

Each annotation is saved as a `.txt` file in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 🔬 Used In

This tool was developed to build the annotation pipeline for the **TÜBİTAK wisdom tooth extraction difficulty prediction project**.

---

## 📄 License

MIT License
