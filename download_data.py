"""
Roboflow'dan veri seti indirme scripti.
Kullanım: API key'i .env dosyasından okur.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv yoksa ortam değişkenlerinden oku

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError(
        "ROBOFLOW_API_KEY ortam değişkeni bulunamadı.\n"
        "Lütfen .env dosyası oluşturup içine şunu yazın:\n"
        "ROBOFLOW_API_KEY=your_api_key_here"
    )

rf = Roboflow(api_key=api_key)
project = rf.workspace("meyve-eyp6i").project("wisdom-teeth-nbnzt-mnmd0")
version = project.version(2)
dataset = version.download("yolov8")
