import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Medicinal_plant_test"
MODEL = ROOT / "ml_v2/results/plant_mobilenetv2/best_plant_mobilenetv2.keras"
MAPPING = ROOT / "ml_v2/results/plant_mobilenetv2/class_indices.json"
OUT = ROOT / "ml_v2/results/plant_mobilenetv2/evaluation"

OUT.mkdir(parents=True, exist_ok=True)

with open(MAPPING) as f:
    class_indices = json.load(f)

classes = [name for name, idx in
           sorted(class_indices.items(), key=lambda item: item[1])]

model = tf.keras.models.load_model(MODEL)

paths, y_true = [], []
valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

for folder in sorted(DATA.iterdir()):
    if not folder.is_dir():
        continue
    if folder.name not in class_indices:
        print("WARNING: Unknown test folder:", folder.name)
        continue
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in valid_ext:
            paths.append(p)
            y_true.append(class_indices[folder.name])

print("Images:", len(paths))
print("Classes:", len(classes))
print("Loading and predicting images...")

y_pred = []
for start in range(0, len(paths), 32):
    batch_paths = paths[start:start + 32]
    images = []
    for p in batch_paths:
        img = tf.keras.utils.load_img(p, target_size=(224, 224))
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        images.append(arr)
    predictions = model.predict(np.asarray(images), verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

print("\nOverall accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
print("Macro-F1:", round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4))

report = classification_report(
    y_true, y_pred, labels=list(range(len(classes))),
    target_names=classes, zero_division=0, output_dict=True
)
print("\nPer-class results:")
for name in classes:
    row = report[name]
    print(f"{name:20s} precision={row['precision']:.3f} "
          f"recall={row['recall']:.3f} F1={row['f1-score']:.3f} "
          f"support={int(row['support'])}")

with open(OUT / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

np.savetxt(OUT / "confusion_matrix.csv",
           confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
           delimiter=",", fmt="%d")

print("\nSaved report to:", OUT)
