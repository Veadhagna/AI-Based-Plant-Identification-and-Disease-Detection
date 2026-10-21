import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ml_v2/data/disease_split/test"
MODEL = ROOT / "ml_v2/results/disease_model/best_disease_model.keras"
LABELS = ROOT / "ml_v2/results/disease_model/class_names.json"
OUT = ROOT / "ml_v2/results/disease_model/evaluation"
OUT.mkdir(parents=True, exist_ok=True)

with open(LABELS) as f:
    classes = json.load(f)

if len(classes) != len(set(classes)):
    raise ValueError("Duplicate labels found in class_names.json")

class_to_idx = {name: i for i, name in enumerate(classes)}
folders = {p.name for p in DATA.iterdir() if p.is_dir()}

if folders != set(classes):
    raise ValueError("Test folders do not match model labels")

model = tf.keras.models.load_model(MODEL)
print("Model input shape:", model.input_shape)
print("Classes:", len(classes))

paths, y_true = [], []
extensions = {".jpg", ".jpeg", ".png", ".webp"}

for class_name in classes:
    folder = DATA / class_name
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p)
            y_true.append(class_to_idx[class_name])

print("Test images:", len(paths))
print("Predicting...")

y_pred = []
for start in range(0, len(paths), 16):
    batch_paths = paths[start:start + 16]
    images = []

    for p in batch_paths:
        img = tf.keras.utils.load_img(p, target_size=(224, 224))
        arr = tf.keras.utils.img_to_array(img)
        # EfficientNetB0 model includes its own preprocessing.
        # Do not divide pixel values by 255 here.
        images.append(arr)

    predictions = model.predict(np.asarray(images), verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print(f"\nOverall accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Correct: {(y_true == y_pred).sum()} / {len(y_true)}")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

print("\nPer-class results:")
for name in classes:
    row = report[name]
    print(
        f"{name:38s} "
        f"precision={row['precision']:.3f} "
        f"recall={row['recall']:.3f} "
        f"F1={row['f1-score']:.3f} "
        f"support={int(row['support'])}"
    )

with open(OUT / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

np.savetxt(
    OUT / "confusion_matrix.csv",
    confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
    delimiter=",",
    fmt="%d",
)

print("\nSaved results to:", OUT)
