from pathlib import Path
from severity_analysis import analyze_severity
import argparse
import json

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

IMAGE_SIZE = (160, 160)


# --------------------------------------------------
# Load model and class names
# --------------------------------------------------

def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    return model, class_names


# --------------------------------------------------
# Load image
# --------------------------------------------------

def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    return image


# --------------------------------------------------
# Disease classification
# --------------------------------------------------

def classify_image(model, class_names, image):
    resized = cv2.resize(image, IMAGE_SIZE)

    rgb_image = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    image_array = rgb_image.astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    disease = class_names[predicted_index]

    return disease, confidence


# --------------------------------------------------
# Severity analysis
# --------------------------------------------------

def estimate_severity(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Green leaf mask
    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])

    leaf_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_pixels = cv2.countNonZero(
        leaf_mask
    )

    if leaf_pixels == 0:
        return 0.0, "Unable to estimate"

    # Detect abnormal regions
    # Brown / yellow / dark areas
    lower_abnormal_1 = np.array([5, 40, 20])
    upper_abnormal_1 = np.array([35, 255, 220])

    lower_abnormal_2 = np.array([0, 0, 0])
    upper_abnormal_2 = np.array([180, 255, 90])

    abnormal_mask_1 = cv2.inRange(
        hsv,
        lower_abnormal_1,
        upper_abnormal_1
    )

    abnormal_mask_2 = cv2.inRange(
        hsv,
        lower_abnormal_2,
        upper_abnormal_2
    )

    abnormal_mask = cv2.bitwise_or(
        abnormal_mask_1,
        abnormal_mask_2
    )

    # Only count abnormal pixels inside the leaf
    abnormal_mask = cv2.bitwise_and(
        abnormal_mask,
        leaf_mask
    )

    abnormal_pixels = cv2.countNonZero(
        abnormal_mask
    )

    affected_percentage = (
        abnormal_pixels / leaf_pixels
    ) * 100

    # Project-defined severity thresholds
    if affected_percentage < 15:
        severity = "Low"
    elif affected_percentage <= 40:
        severity = "Moderate"
    else:
        severity = "High"

    return affected_percentage, severity


# --------------------------------------------------
# Generate recommendation
# --------------------------------------------------

def generate_recommendation(
    disease,
    severity
):
    if disease == "healthy":
        return (
            "The leaf appears healthy based on the "
            "classification result."
        )

    if severity == "Low":
        return (
            "A low visual severity was estimated. "
            "Continue monitoring the plant."
        )

    if severity == "Moderate":
        return (
            "A moderate visual severity was estimated. "
            "Further inspection of the plant is recommended."
        )

    if severity == "High":
        return (
            "A high visual severity was estimated. "
            "Further inspection and appropriate plant-care "
            "guidance should be considered."
        )

    return (
        "Review the image and classification result."
    )


# --------------------------------------------------
# Main prediction pipeline
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Plant Disease Detection and Severity Analysis"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the leaf image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"\nERROR: Image not found: {image_path}")
        return

    print("\nLoading model...")
    model, class_names = load_resources()

    print("Loading image...")
    image = load_image(image_path)

    print("Classifying disease...")
    disease, confidence = classify_image(
        model,
        class_names,
        image
    )

    print("Estimating severity...")

    affected_percentage, severity, visualization_path = analyze_severity(
        image
    )

    recommendation = generate_recommendation(
        disease,
        severity
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PLANT DISEASE DETECTION RESULT")
    print("=" * 60)

    print(f"Disease       : {disease}")
    print(f"Confidence    : {confidence * 100:.2f}%")

    if severity == "Unable to estimate":
        print("Severity      : Unable to estimate")
        print("Affected Area : N/A")
    else:
        print(f"Severity      : {severity}")
        print(f"Visual Result : {visualization_path}")
        print(
            f"Affected Area : {affected_percentage:.2f}%"
        )

    print("\nRecommendation:")
    print(recommendation)

    print("=" * 60)
    print(
        "\nNote: Severity is a project-defined "
        "computer-vision estimate."
    )


if __name__ == "__main__":
    main()