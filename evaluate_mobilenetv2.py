import os
import json
import hashlib
import numpy as np
import tensorflow as tf

from PIL import Image
from collections import defaultdict
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

TRAIN_DIR = "Medicinal_plant_dataset_Train"
TEST_DIR = "Medicinal_plant_test"
MODEL_PATH = "ml_v2/results/plant_mobilenetv2/best_plant_mobilenetv2.keras"
MAPPING_PATH = "ml_v2/results/plant_mobilenetv2/class_indices.json"

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Load class mapping created during training
with open(MAPPING_PATH) as f:
    class_indices = json.load(f)

class_names = [
    name for name, index in
    sorted(class_indices.items(), key=lambda item: item[1])
]

# Hash training images to identify exact duplicates
train_hashes = set()

for current, _, files in os.walk(TRAIN_DIR):
    for filename in files:
        if filename.lower().endswith(EXTENSIONS):
            path = os.path.join(current, filename)
            with open(path, "rb") as f:
                train_hashes.add(hashlib.sha256(f.read()).hexdigest())

# Collect test images excluding exact duplicates
samples = []
excluded = 0
unknown_classes = []

for class_name in sorted(os.listdir(TEST_DIR)):
    folder = os.path.join(TEST_DIR, class_name)

    if not os.path.isdir(folder):
        continue

    if class_name not in class_indices:
        unknown_classes.append(class_name)
        continue

    for current, _, files in os.walk(folder):
        for filename in files:
            if not filename.lower().endswith(EXTENSIONS):
                continue

            path = os.path.join(current, filename)

            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()

            if digest in train_hashes:
                excluded += 1
                continue

            samples.append((path, class_indices[class_name]))

if unknown_classes:
    raise ValueError(f"Unknown test classes: {unknown_classes}")

if not samples:
    raise ValueError("No test images found.")

print("Exact duplicate test images excluded:", excluded)
print("Clean test images:", len(samples))

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

correct = 0
total = 0
per_class = defaultdict(lambda: [0, 0])

for start in range(0, len(samples), BATCH_SIZE):
    batch = samples[start:start + BATCH_SIZE]
    images = []

    for path, _ in batch:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            images.append(np.asarray(image, dtype=np.float32))

    # Required MobileNetV2 preprocessing
    x = preprocess_input(np.array(images))
    predictions = model.predict(x, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    for (_, true_label), pred_label in zip(batch, predicted_labels):
        total += 1
        per_class[class_names[true_label]][1] += 1

        if int(pred_label) == true_label:
            correct += 1
            per_class[class_names[true_label]][0] += 1

print(f"\nClean test accuracy: {correct}/{total} = {100 * correct / total:.2f}%")

print("\nPer-class accuracy:")
for name in class_names:
    right, count = per_class[name]
    if count:
        print(f"{name}: {right}/{count} ({100 * right / count:.1f}%)")
