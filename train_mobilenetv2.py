import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = "Medicinal_plant_dataset_Train"
OUTPUT_DIR = "ml_v2/results/plant_mobilenetv2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

# Training and validation use the same class mapping and a fixed split.
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False
)

if train_data.class_indices != val_data.class_indices:
    raise RuntimeError("Training and validation class mappings differ.")

num_classes = train_data.num_classes
print("Classes:", num_classes)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

# Load pretrained MobileNetV2 feature extractor.
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

best_path = os.path.join(OUTPUT_DIR, "best_plant_mobilenetv2.keras")

callbacks = [
    ModelCheckpoint(
        best_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nStarting MobileNetV2 training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

model.save(os.path.join(OUTPUT_DIR, "plant_mobilenetv2_final.keras"))

print("\nTraining finished.")
print("Best checkpoint:", best_path)
print("Final model:", os.path.join(OUTPUT_DIR, "plant_mobilenetv2_final.keras"))
print("Class mapping:", os.path.join(OUTPUT_DIR, "class_indices.json"))
