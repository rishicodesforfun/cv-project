import cv2
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_leaf_mask(image):
    """
    Segment the main leaf from the background using GrabCut.
    """

    height, width = image.shape[:2]

    # Start with the whole image as probable background
    mask = np.zeros(
        (height, width),
        np.uint8
    )

    # Leave a small border around the image as background.
    border = 5

    mask[:] = cv2.GC_PR_BGD
    mask[
        border:height-border,
        border:width-border
    ] = cv2.GC_PR_FGD

    # Use a rectangle around the central object.
    rect = (
        max(1, int(width * 0.05)),
        max(1, int(height * 0.05)),
        int(width * 0.90),
        int(height * 0.90)
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Keep definite/probable foreground
    leaf_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Remove small noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

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

    # Keep the largest connected region
    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            leaf_mask
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        leaf_mask = clean_mask

    return leaf_mask


def create_disease_mask(image, leaf_mask):
    """
    Detect visually abnormal regions inside the leaf.

    This is a project-defined computer-vision heuristic.
    It is not a scientifically validated disease-severity
    standard.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------
    # Brown / yellow regions
    # --------------------------------------------------

    lower_brown = np.array([
        5, 45, 20
    ])

    upper_brown = np.array([
        38, 255, 230
    ])

    brown_mask = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown
    )

    # --------------------------------------------------
    # Dark spots
    # --------------------------------------------------

    lower_dark = np.array([
        0, 0, 0
    ])

    upper_dark = np.array([
        180, 255, 65
    ])

    dark_mask = cv2.inRange(
        hsv,
        lower_dark,
        upper_dark
    )

    # --------------------------------------------------
    # Combine abnormal regions
    # --------------------------------------------------

    disease_mask = cv2.bitwise_or(
        brown_mask,
        dark_mask
    )

    # Only count pixels inside the leaf
    disease_mask = cv2.bitwise_and(
        disease_mask,
        leaf_mask
    )

    # Remove tiny isolated regions
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return disease_mask


def calculate_severity(
    leaf_mask,
    disease_mask
):
    """
    Calculate visually affected leaf area.
    """

    leaf_pixels = cv2.countNonZero(
        leaf_mask
    )

    disease_pixels = cv2.countNonZero(
        disease_mask
    )

    if leaf_pixels == 0:
        return 0.0, "Unable to estimate"

    affected_percentage = (
        disease_pixels /
        leaf_pixels
    ) * 100

    # Project-defined thresholds
    if affected_percentage < 15:
        severity = "Low"

    elif affected_percentage <= 40:
        severity = "Moderate"

    else:
        severity = "High"

    return affected_percentage, severity


def create_visualization(
    image,
    leaf_mask,
    disease_mask,
    output_path
):
    """
    Create the final visual result.
    """

    # Create a copy of the original
    result = image.copy()

    # Highlight abnormal regions
    highlight = np.zeros_like(
        image
    )

    highlight[
        disease_mask > 0
    ] = (
        0,
        0,
        255
    )

    result = cv2.addWeighted(
        result,
        0.70,
        highlight,
        0.30,
        0
    )

    # Draw leaf boundary
    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        result,
        contours,
        -1,
        (255, 255, 255),
        2
    )

    # Save masks
    cv2.imwrite(
        str(OUTPUT_DIR / "leaf_mask.jpg"),
        leaf_mask
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "disease_mask.jpg"),
        disease_mask
    )

    # Save final visualization
    cv2.imwrite(
        str(output_path),
        result
    )


def analyze_severity(image):
    """
    Complete severity-analysis pipeline.
    """

    leaf_mask = create_leaf_mask(
        image
    )

    disease_mask = create_disease_mask(
        image,
        leaf_mask
    )

    affected_percentage, severity = (
        calculate_severity(
            leaf_mask,
            disease_mask
        )
    )

    output_path = (
        OUTPUT_DIR /
        "severity_result.jpg"
    )

    create_visualization(
        image,
        leaf_mask,
        disease_mask,
        output_path
    )

    return (
        affected_percentage,
        severity,
        output_path
    )