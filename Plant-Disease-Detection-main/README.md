# Plant Disease Detection and Severity Analysis Using Computer Vision

## 1. Project Overview

This project is a computer vision-based system for detecting diseases in tomato plant leaves and estimating the visual severity of the detected disease.

The system accepts a tomato leaf image as input and performs image validation, preprocessing, disease classification, visual severity analysis, and result generation.

The disease classification model uses transfer learning with MobileNetV2.

---

## 2. Problem Statement

Identifying plant diseases from leaf images manually can be difficult and time-consuming. This project aims to develop a computer vision system that can classify common tomato leaf diseases from images and provide an estimated visual severity level.

---

## 3. Objectives

- Detect diseases from tomato leaf images.
- Classify the input image into one of 10 tomato disease/health categories.
- Estimate the visually affected leaf area.
- Categorize visual severity as Low, Moderate, or High.
- Generate an easy-to-understand prediction result.
- Demonstrate the application of computer vision and deep learning techniques.

---

## 4. Main Features

1. Image input and validation
2. Image preprocessing
3. Tomato leaf disease classification
4. Visual severity analysis
5. Affected area estimation
6. Result and recommendation generation
7. Model evaluation and testing

---

## 5. System Workflow

```text
Leaf Image
    |
    v
Image Validation
    |
    v
Image Preprocessing
    |
    v
Disease Classification
    |
    v
Severity Analysis
    |
    v
Result Generation
    |
    v
Final Output
```

---

## 6. Functional Modules

Image Input and Validation:
Accepts the leaf image and verifies that the input can be processed.

Image Preprocessing:
Resizes and prepares the image for the classification model.

Disease Classification:
Uses a trained MobileNetV2-based deep learning model to classify the tomato leaf.

Severity Analysis:
Uses computer vision techniques to estimate the visually affected portion of the leaf.

Result Generation:
Combines the disease prediction, confidence, severity, affected area, and recommendation into a final result.

---

## 7. Disease Classes:
The model supports the following 10 tomato categories:
Tomato Healthy
Tomato Bacterial Spot
Tomato Early Blight
Tomato Late Blight
Tomato Leaf Mold
Tomato Septoria Leaf Spot
Tomato Spider Mites
Tomato Target Spot
Tomato Mosaic Virus
Tomato Yellow Leaf Curl Virus

---

## 8. Technology Stack
Python
TensorFlow / Keras
MobileNetV2
OpenCV
NumPy
Pandas
Matplotlib
Pillow
scikit-learn
Git
GitHub

---

## 9. Project Structure
Plant-Disease-Detection/
|
├── models/
│   ├── class_names.json
│   └── plant_disease_model.keras
|
├── outputs/
│   ├── disease_mask.jpg
│   ├── leaf_mask.jpg
│   └── severity_result.jpg
|
├── src/
│   ├── config.py
│   ├── disease_classifier.py
│   ├── evaluate.py
│   ├── find_demo_image.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── result_generator.py
│   ├── severity_analysis.py
│   ├── train.py
│   └── utils.py
|
├── tests/
│   └── test_prediction.py
|
├── README.md
├── requirements.txt
├── statement.md
└── .gitignore

---

## 10. Installation
Clone the repository and open the project directory.

Create and activate a Python virtual environment:
python -m venv venv

Windows PowerShell:
.\venv\Scripts\Activate.ps1

Install the required dependencies:
pip install -r requirements.txt

---

## 11. Dataset
The project uses the PlantVillage dataset and works with 10 tomato classes.
The dataset is not included in this repository because of its size.

The local dataset is organized into:
data/tomato/
├── train/
├── val/
└── test/

The dataset should be prepared before training using:
python src/prepare_dataset.py

---

## 12. Training the Model
The model can be trained using:
python src/train.py

The trained model is saved as:
models/plant_disease_model.keras

Class names are stored in:
models/class_names.json

---

## 13. Disease Prediction

To predict a disease from an image:
python src/predict.py --image "path/to/leaf_image.jpg"

Example:
python src/predict.py --image "data/tomato/test/late_blight/late_blight_00253.jpg"

The program displays:
Disease
Confidence
Severity
Affected Area
Recommendation
Visual result path

---

## 14. Model Evaluation

The trained model can be evaluated using:
python src/evaluate.py

Evaluation includes:
Test accuracy
Test loss
Precision
Recall
F1-score
Classification report
Confusion matrix
Evaluation Results

On the held-out test set:

Metric	Result
Test Images	2,734
Number of Classes	10
Test Accuracy	89.03%
Test Loss	0.3363
Macro F1-score	85.93%
Weighted F1-score	88.93%

---

## 15. Severity Analysis
The severity module uses computer vision processing to estimate the visually affected region of the leaf.

The process includes:
Leaf segmentation
Identification of visually abnormal regions
Affected pixel calculation
Affected area percentage calculation
Severity categorization

The project uses the following prototype thresholds:
Affected Area	Visual Severity
Less than 15%	Low
15% to 40%	Moderate
More than 40%	High

These thresholds are defined for this project as a visual prototype and should not be interpreted as an agricultural diagnosis or treatment standard.

---

## 16. Example Result

Example demonstration output:

PLANT DISEASE DETECTION RESULT
Disease       : late_blight
Confidence    : 70.45%
Severity      : Moderate
Affected Area : 28.37%

Recommendation:
A moderate visual severity was estimated.
Further inspection of the plant is recommended.

---

## 17. Testing

The project contains automated tests in:
tests/test_prediction.py

Run the tests using:
python -m unittest discover -s tests -v

The tests verify:
Model file availability
Class-name file availability
Correct number of classes
Model prediction output shape

---

## 18. Limitations
The model is trained for the selected tomato classes only.
Prediction performance depends on image quality and similarity to the training data.
The severity module provides a project-defined visual estimate.
Severity estimation has not been presented as a scientifically validated agricultural measurement.
The system should therefore be treated as a computer vision prototype.

---

## 19. Future Enhancements
Support additional crops and diseases.
Improve disease classification accuracy.
Use a larger and more diverse dataset.
Improve leaf and disease-region segmentation.
Develop a web or mobile interface.
Add real-time camera-based detection.
Develop a more extensively validated severity estimation method.

---

## 20. Conclusion
This project demonstrates the use of computer vision and deep learning for tomato plant disease classification and visual severity estimation.
The system combines image preprocessing, transfer learning, disease classification, OpenCV-based severity analysis, automated testing, and structured result generation into a modular computer vision pipeline.


