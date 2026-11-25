import numpy as np
from PIL import Image

from src.smartagro.data import collect_image_dataset, compute_group_ids, check_split_leakage
from src.smartagro.metrics import compute_classification_metrics


def test_compute_group_ids_groups_near_duplicates(tmp_path):
    p1 = tmp_path / "leaf1.jpg"
    p2 = tmp_path / "leaf2.jpg"
    p3 = tmp_path / "leaf3.jpg"

    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(p1)
    img.save(p2)

    checker = np.zeros((32, 32, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    Image.fromarray(checker).save(p3)

    groups = compute_group_ids([str(p1), str(p2), str(p3)], hamming_threshold=4)
    assert len(groups) == 3
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_check_split_leakage_detects_overlap():
    groups = np.array([0, 0, 1, 1, 2, 2])
    train = np.array([0])
    val = np.array([2])
    test = np.array([4])
    assert check_split_leakage(train, val, test, groups) is False

    train2 = np.array([0, 1])
    val2 = np.array([2])
    test2 = np.array([3])
    assert check_split_leakage(train2, val2, test2, groups) is True


def test_classification_metrics_are_numeric_and_ordered():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred, classes=["c0", "c1"])
    assert set(metrics.keys()) >= {"accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_collect_image_dataset_reads_visible_files(tmp_path):
    cls_a = tmp_path / "class_a"
    cls_b = tmp_path / "class_b"
    cls_a.mkdir(); cls_b.mkdir()
    (cls_a / "img1.jpg").write_bytes(b"fake")
    (cls_a / "img2.png").write_bytes(b"fake")
    (cls_b / "img3.jpeg").write_bytes(b"fake")
    classes, paths, labels = collect_image_dataset(str(tmp_path))
    assert classes == ["class_a", "class_b"]
    assert len(paths) == 3
    assert set(labels.tolist()) == {0, 1}
