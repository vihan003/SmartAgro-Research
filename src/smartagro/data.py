import glob
import os
from typing import List, Tuple

import numpy as np


def collect_image_dataset(root: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Collects image files grouped by folder name in a dataset root."""
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    paths: List[str] = []
    labels: List[int] = []
    for idx, label_name in enumerate(classes):
        class_dir = os.path.join(root, label_name)
        for path in sorted(glob.glob(os.path.join(class_dir, "*"))):
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(path)
                labels.append(idx)
    return classes, np.asarray(paths), np.asarray(labels, dtype=np.int32)


def compute_group_ids(paths: list, hamming_threshold: int = 4) -> np.ndarray:
    """Computes approximate duplicate groups by dHash similarity."""
    from PIL import Image
    import imagehash

    if len(paths) == 0:
        return np.asarray([], dtype=np.int32)

    hashes = []
    for path in paths:
        img = Image.open(path).convert("L")
        hashes.append(int(str(imagehash.dhash(img)), 16))

    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            dist = bin(hashes[i] ^ hashes[j]).count("1")
            if dist <= hamming_threshold:
                union(i, j)

    return np.asarray([find(i) for i in range(len(paths))], dtype=np.int32)


def check_split_leakage(train_idx, val_idx, test_idx, groups) -> bool:
    """Returns True if any duplicate group appears in multiple splits."""
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    return bool((train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups))
