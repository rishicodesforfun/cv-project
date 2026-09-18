# Project Statement

## Project Title

Plant Disease Detection and Severity Analysis Using Computer Vision

## Problem Statement

Identifying plant diseases from leaf images manually can be difficult and time-consuming. Early identification of visible disease symptoms can help in understanding the condition of a plant.

This project develops a computer vision-based system that accepts a tomato leaf image, classifies it into one of the supported disease or healthy categories, and estimates the visually affected portion of the leaf.

## Scope

The project focuses on computer vision and deep learning techniques for tomato leaf disease classification.

The system supports 10 tomato leaf categories from the selected PlantVillage dataset.

The project also includes a computer vision-based prototype for estimating the visually affected leaf area and categorizing it into Low, Moderate, or High visual severity.

The severity estimation is a project-defined visual estimate and is not intended to represent a scientifically validated agricultural diagnosis.

## Target Users

- Students and learners studying computer vision and machine learning
- Researchers and developers working on plant disease image classification
- Users interested in experimenting with computer vision-based plant analysis

## High-Level Features

1. Tomato leaf image input and validation
2. Image preprocessing
3. Disease classification using a MobileNetV2-based deep learning model
4. Confidence score generation
5. Visual severity estimation
6. Affected leaf area calculation
7. Result and recommendation generation
8. Model evaluation and automated testing

## Supported Disease Categories

The system supports the following 10 categories:

1. Tomato Healthy
2. Tomato Bacterial Spot
3. Tomato Early Blight
4. Tomato Late Blight
5. Tomato Leaf Mold
6. Tomato Septoria Leaf Spot
7. Tomato Spider Mites
8. Tomato Target Spot
9. Tomato Mosaic Virus
10. Tomato Yellow Leaf Curl Virus

## Expected Output

For a given tomato leaf image, the system produces:

- Predicted disease category
- Prediction confidence
- Estimated visual severity
- Estimated affected area percentage
- A project-defined recommendation
- Visual severity analysis output