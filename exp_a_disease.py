"""Experiment A: crop-disease classification on PlantVillage with near-duplicate GROUPED splits.

Why grouped: PlantVillage has several near-identical shots per leaf and no public leaf IDs, so a random split leaks.
We build groups by perceptual-hash clustering (dHash, Hamming distance <= --hamming) and use StratifiedGroupKFold.
We also run a deliberately LEAKY random split with the same model, to measure how much leakage inflates the score.

DATA: put PlantVillage (unaugmented, one folder per class) in data/plantvillage/<class>/*.jpg
      (e.g. the Mendeley copy https://data.mendeley.com/datasets/tywbtsjrjv/1, or `tfds` export). Check its licence yourself.
RUN (Colab GPU is fine):  pip install tensorflow scikit-learn pillow imagehash pandas matplotlib
      python exp_a_disease.py --data data/plantvillage --out results_a --img 128 --epochs 6 --seed 0
Nothing here has been run by the author of this file; results come only from YOUR run.
"""
import argparse, json, os, time, platform, glob
import numpy as np, pandas as pd

def collect(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in glob.glob(os.path.join(root, c, "*")):
            if p.lower().endswith((".jpg", ".jpeg", ".png")): paths.append(p); labels.append(i)
    return classes, np.array(paths), np.array(labels)

def near_duplicate_groups(paths, hamming=4):
    """Union-find over dHash; 5 hash chunks so any pair within Hamming<=4 shares at least one chunk (pigeonhole)."""
    from PIL import Image; import imagehash
    h = [int(str(imagehash.dhash(Image.open(p).convert("L"))), 16) for p in paths]
    parent = list(range(len(paths)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    cuts = [(0, 12), (12, 25), (25, 38), (38, 51), (51, 64)]
    for lo, hi in cuts:
        buckets = {}
        for i, v in enumerate(h): buckets.setdefault((v >> lo) & ((1 << (hi - lo)) - 1), []).append(i)
        for b in buckets.values():
            for a in range(len(b)):
                for c in range(a + 1, len(b)):
                    if bin(h[b[a]] ^ h[b[c]]).count("1") <= hamming: parent[find(b[a])] = find(b[c])
    return np.array([find(i) for i in range(len(paths))])

def make_ds(tf, paths, labels, img, batch, train):
    def load(p, y):
        x = tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(p), channels=3), (img, img)); return x, y
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train: ds = ds.shuffle(2048, seed=0)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(tf, kind, n, img):
    L = tf.keras.layers; aug = tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(.1), L.RandomBrightness(.15)])
    inp = tf.keras.Input((img, img, 3)); x = aug(inp)
    if kind == "small_cnn":
        x = L.Rescaling(1 / 255)(x)
        for f in (32, 64, 128): x = L.MaxPool2D()(L.Conv2D(f, 3, activation="relu")(x))
        x = L.Dense(128, activation="relu")(L.GlobalAveragePooling2D()(x))
    else:
        base = tf.keras.applications.MobileNetV2((img, img, 3), include_top=False, weights="imagenet"); base.trainable = False
        x = L.GlobalAveragePooling2D()(base(tf.keras.applications.mobilenet_v2.preprocess_input(x), training=False))
    m = tf.keras.Model(inp, L.Dense(n, activation="softmax")(L.Dropout(.2)(x)))
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"]); return m

def run(a):
    import tensorflow as tf
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tf.keras.utils.set_random_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    classes, paths, y = collect(a.data); print(len(paths), "images,", len(classes), "classes")
    groups = near_duplicate_groups(paths, a.hamming); print(len(set(groups)), "near-duplicate groups")
    report = {"n_images": int(len(paths)), "n_classes": len(classes), "n_groups": int(len(set(groups))), "seed": a.seed,
              "img": a.img, "epochs": a.epochs, "hamming": a.hamming, "tensorflow": tf.__version__, "python": platform.python_version(),
              "class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)}, "runs": {}}
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(paths, y, groups))
    splits = {"grouped": (np.concatenate([folds[i][1] for i in (0, 1, 2)]), folds[3][1], folds[4][1])}     # train / val / test by group
    rs = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(paths, y))
    splits["random_LEAKY"] = (np.concatenate([rs[i][1] for i in (0, 1, 2)]), rs[3][1], rs[4][1])
    assert not set(groups[splits["grouped"][0]]) & set(groups[splits["grouped"][2]]), "group leakage train/test"
    for sname, (tr, va, te) in splits.items():
        for kind in ("small_cnn", "mobilenetv2"):
            tag = f"{sname}__{kind}"; t0 = time.time(); m = build_model(tf, kind, len(classes), a.img)
            m.fit(make_ds(tf, paths[tr], y[tr], a.img, 32, True), validation_data=make_ds(tf, paths[va], y[va], a.img, 32, False), epochs=a.epochs, verbose=2)
            prob = m.predict(make_ds(tf, paths[te], y[te], a.img, 32, False), verbose=0); pred = prob.argmax(1)
            lat = []; x1 = tf.zeros((1, a.img, a.img, 3))
            for _ in range(30): s = time.time(); m(x1, training=False); lat.append(time.time() - s)
            report["runs"][tag] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "accuracy": float(accuracy_score(y[te], pred)),
                                   "macro_f1": float(f1_score(y[te], pred, average="macro")), "params": int(m.count_params()),
                                   "cpu_or_gpu_latency_ms_median": float(np.median(lat) * 1000), "train_seconds": round(time.time() - t0, 1)}
            pd.DataFrame(classification_report(y[te], pred, target_names=classes, output_dict=True, zero_division=0)).T.to_csv(f"{a.out}/per_class_{tag}.csv")
            cm = confusion_matrix(y[te], pred); plt.figure(figsize=(9, 8)); plt.imshow(cm, cmap="Blues"); plt.title(tag); plt.tight_layout(); plt.savefig(f"{a.out}/cm_{tag}.png"); plt.close()
            print(tag, report["runs"][tag])
    json.dump(report, open(f"{a.out}/results_a.json", "w"), indent=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", default="results_a"); p.add_argument("--img", type=int, default=128)
    p.add_argument("--epochs", type=int, default=6); p.add_argument("--seed", type=int, default=0); p.add_argument("--hamming", type=int, default=4); run(p.parse_args())
