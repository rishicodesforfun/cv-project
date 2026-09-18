from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf


TEST_DIR = Path("data/tomato/test")
MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

IMAGE_SIZE = (160, 160)


with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

model = tf.keras.models.load_model(MODEL_PATH)


best_result = None


for class_name in class_names:

    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        continue

    for image_path in class_dir.glob("*.jpg"):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        prediction = model.predict(
            image,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]

        confidence = float(
            prediction[predicted_index]
        )

        # Only consider correctly classified images
        if predicted_class == class_name:

            if best_result is None or confidence > best_result[0]:

                best_result = (
                    confidence,
                    class_name,
                    image_path
                )

    if best_result is not None:
        print(
            f"Current best: "
            f"{best_result[1]} - "
            f"{best_result[0] * 100:.2f}%"
        )


if best_result is None:

    print("\nNo correctly classified image found.")

else:

    confidence, actual_class, image_path = best_result

    print("\n" + "=" * 60)
    print("BEST DEMONSTRATION IMAGE")
    print("=" * 60)

    print(f"Actual class : {actual_class}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Image        : {image_path}")

    print("=" * 60)