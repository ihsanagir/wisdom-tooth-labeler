import zipfile

import pytest

import train_pose
from geometry import KEYPOINTS
from tests.conftest import SAMPLE_DIR


def pose_line(n_points):
    pts = ["0.5 0.5 2"] * n_points + ["0 0 0"] * (len(KEYPOINTS) - n_points)
    return "0 0.5 0.5 0.1 0.2 " + " ".join(pts)


def test_count_points():
    assert train_pose.count_points(pose_line(3) + "\n" + pose_line(2)) == 5
    assert train_pose.count_points(pose_line(0)) == 0


def test_prepare_builds_split_and_yaml(tmp_path, monkeypatch):
    images = sorted(SAMPLE_DIR.glob("*.jpg"))[:6]
    if len(images) < 6:
        pytest.skip("dataset_v2 bulunamadı")

    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i, img in enumerate(images):
            z.write(img, f"dataset/images/{img.name}")
            # son görüntüde hiç nokta yok → veri setine alınmamalı
            z.writestr(f"dataset/labels_pose/{img.stem}.txt", pose_line(0 if i == 5 else 4))

    monkeypatch.setattr(train_pose, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(train_pose, "YAML_PATH", tmp_path / "pose.yaml")
    train_pose.prepare(zpath)

    copied = list((tmp_path / "out").glob("*/images/*.jpg"))
    assert len(copied) == 5
    assert all((p.parent.parent / "labels" / f"{p.stem}.txt").exists() for p in copied)
    yaml = (tmp_path / "pose.yaml").read_text(encoding="utf-8")
    assert f"kpt_shape: [{len(KEYPOINTS)}, 3]" in yaml
