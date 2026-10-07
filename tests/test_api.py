import io
import zipfile

from geometry import KEYPOINTS

LABEL = {
    "impaction": "Mesioangular",
    "ramus": "Sınıf 1 (Önünde)",
    "depth": "Seviye A (Oklüzal)",
}


def save(client, name, idx, **extra):
    body = {"image_name": name, "bbox_index": idx, "bbox": [100, 100, 200, 260], **LABEL, **extra}
    return client.post("/api/label/save", json=body)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_save_with_keypoints_and_stable_indices_after_delete(client, sample_image):
    name = sample_image.name
    kp = {"m3_crown": [150, 110], "m3_apex": [150, 250], "unknown_point": [1, 1]}
    assert save(client, name, 1).json()["status"] == "saved"
    assert save(client, name, 3, keypoints=kp, jaw="Alt Çene").json()["status"] == "saved"

    client.post("/api/label/delete_box", json={"image_name": name, "bbox_index": 1})
    labels = client.get("/api/label/existing", params={"image_name": name}).json()["labels"]

    # Silme sonrası kalan etiketin indeksi değişmemeli (eski hata: 3 → 1 oluyordu)
    assert [l["bbox_index"] for l in labels] == [3]
    assert labels[0]["keypoints"] == {"m3_crown": [150, 110], "m3_apex": [150, 250]}
    assert labels[0]["jaw"] == "Alt Çene"


def test_export_contains_pose_labels(client, sample_image):
    save(client, sample_image.name, 3, keypoints={"m3_crown": [150, 110]}, jaw="Alt Çene")
    assert client.get("/api/label/export").status_code == 401

    r = client.get("/api/label/export", headers={"X-Admin-Token": "test-token"})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert "dataset/data_pose.yaml" in names
    pose = z.read(f"dataset/labels_pose/{sample_image.stem}.txt").decode().split()
    assert len(pose) == 5 + 3 * len(KEYPOINTS)
    # ilk nokta (m3_crown) işaretli, ikincisi işaretsiz
    assert pose[7] == "2" and pose[8:11] == ["0", "0", "0"]
    assert f"kpt_shape: [{len(KEYPOINTS)}, 3]" in z.read("dataset/data_pose.yaml").decode()


def test_geometry_endpoint(client):
    kp = {"m3_crown": [220, 400], "m3_apex": [220, 520], "m2_apex": [310, 520],
          "m2_cusp_mesial": [330, 400], "m2_cusp_distal": [290, 400]}
    r = client.post("/api/label/geometry", json={"keypoints": kp, "jaw": "Alt Çene"}).json()
    assert r["impaction"]["value"] == "Dikey (Vertical)"
    assert "canal_top" in r["missing"]

    schema = client.get("/api/label/keypoints").json()
    assert [k["name"] for k in schema["keypoints"]] == KEYPOINTS
    assert "canal_top" not in schema["upper_jaw"]


def test_path_traversal_rejected(client):
    assert client.get("/api/label/detect", params={"image_name": "../app.py"}).status_code == 404
    assert save(client, "../../x.jpg", 1).status_code == 400


def test_analyze_rejects_unknown_option(client):
    body = {"gender": "Erkek", "age": "<20", "mouth_opening": "Normal (>40mm)",
            "impaction": "Bilinmeyen", "ramus": "Sınıf 1 (Önünde)", "depth": "Seviye A (Oklüzal)",
            "root": "Normal/Konik", "nerve": "Uzak"}
    assert client.post("/api/analyze", json=body).status_code == 422


def test_admin_requires_token(client):
    assert client.get("/api/admin/images").status_code == 401
    assert client.get("/api/admin/images", headers={"X-Admin-Token": "test-token"}).status_code == 200
