from pathlib import Path
import json

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_DIR = Path("data/tomato/test")
MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32


# --------------------------------------------------
# Load class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

print("\nEvaluating model...")
print("=" * 60)

loss, accuracy = model.evaluate(test_dataset, verbose=1)

print("\nTest Results")
print("-" * 60)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

true_labels = []
predicted_labels = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)

    predicted = tf.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted.numpy())


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix")
print("=" * 60)
print(matrix)

print("\nEvaluation completed successfully.")