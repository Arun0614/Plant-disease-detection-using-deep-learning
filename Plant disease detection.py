
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Basic Configuration

BASE_DIR = "New Plant Diseases Dataset(Small)"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR = os.path.join(BASE_DIR, "test")

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 5
MODEL_FILENAME = "plant_mobilenetv2.keras"

# Verify dataset folders
assert os.path.isdir(TRAIN_DIR), f"Train directory not found: {TRAIN_DIR}"
assert os.path.isdir(VALID_DIR), f"Valid directory not found: {VALID_DIR}"

# GPU check
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# Dataset Preparation

print("Preparing datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="categorical",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, labels="inferred", label_mode="categorical",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names[:10]}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Data Augmentation

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# Model Creation

print("Building MobileNetV2 model...")

base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze base

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()


# Training Callbacks

callbacks = [
    ModelCheckpoint(MODEL_FILENAME, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]


# Training

print("\nStarting training...")
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)


# Evaluation

print("\nEvaluating on validation set...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

# Save final model
model.save(MODEL_FILENAME)
print(f"Model saved as {MODEL_FILENAME}")


# Prediction Function

def predict_image(img_path, model_path=MODEL_FILENAME, top_k=3):
    """Predict disease name and show image with label."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)[0]

    top_idx = preds.argsort()[-top_k:][::-1]
    results = [(class_names[i], float(preds[i])) for i in top_idx]

    # Display image with top prediction
    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    title = f"Predicted: {results[0][0]} ({results[0][1]*100:.2f}%)"
    plt.title(title, color="green", fontsize=12, weight="bold")
    plt.show()
    return results


# Run a Prediction Example

if __name__ == "__main__":
    test_image = "/Users/arun/PycharmProjects/PythonProject/New Plant Diseases Dataset(Small)/train/Apple___Black_rot/fbd0bf3e-2260-4a97-be8a-af9cac60df7b___JR_FrgE.S 2807.JPG"

    print("\nPredicting for image:", test_image)
    results = predict_image(test_image)
    print("\nTop Predictions:")
    for name, prob in results:
        print(f"  {name}: {prob*100:.2f}%")
