import os
import shutil
import tempfile
from pathlib import Path

import pytest

# app ve label_storage klasörleri import anında okur → testler geçici klasörlerde çalışır
_TMP = Path(tempfile.mkdtemp(prefix="yirmilik_test_"))
os.environ["IMAGES_DIR"] = str(_TMP / "images")
os.environ["LABELS_DIR"] = str(_TMP / "labels")
os.environ.pop("APP_USER", None)
os.environ.pop("APP_PASSWORD", None)
os.environ["ADMIN_TOKEN"] = "test-token"

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "dataset_v2" / "test" / "images"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import app
    return TestClient(app.app)


@pytest.fixture(scope="session")
def sample_image():
    """Testler için images klasörüne bir örnek röntgen kopyalar."""
    images = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not images:
        pytest.skip("dataset_v2 bulunamadı (önce veri_bol.py çalıştırın)")
    dest = Path(os.environ["IMAGES_DIR"]) / "ornek.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(images[0], dest)
    return dest


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
