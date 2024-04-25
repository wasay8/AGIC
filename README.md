```markdown
# Automated Garbage Image Classifier (AGIC)

## Overview

AGIC is a web-based application developed to improve waste management by classifying waste images into 6 broad categories: Plastic, Metal, Paper, Cardboard, Glass, and Trash. The classifier also determines the disposability and sustainability status of each item as Biodegradable or Non-Biodegradable. 

## Key Features

- **High Accuracy Classification**
  - Achieved 90.89% and 93.26% accuracy using ResNet-34 and ResNet-50 architectures respectively.
  
- **Model Evaluation**
  - Implemented a systematic approach to model evaluation including visualization of loss function, learning rate optimization, and confusion matrix analysis.
  
- **Sustainability Focus**
  - Addresses the growing problem of waste management by distinguishing between Biodegradable and Non-Biodegradable waste, promoting sustainable disposal and recycling.

## Dataset

- **Source**: Combination of 2527 good quality JPG images.
- **Categories**: 
  - Glass: 501
  - Paper: 594
  - Cardboard: 403
  - Plastic: 482
  - Metal: 410
  - Trash: 137

## Model Architecture

- **Convolutional Neural Networks (CNN)**
  - Utilized ResNet-34 and ResNet-50 architectures for hyperparameter tuning.
  
## Libraries and Frameworks

- **Fastai**: Built on top of PyTorch, aiding in building complex model structures using Deep Learning algorithms with transfer learning.
  
- **NumPy**: Used for array operations and mathematical computations.

## Experimental Analysis

- **Training and Validation**
  - Data split: 80% training, 20% validation
  - Batch size: 16
  - Learning rate optimization and hyperparameter tuning
  
- **Evaluation Metrics**
  - Confusion matrix and F1 score for model assessment
  
## Conclusion

AGIC serves as an efficient tool for waste classification and sustainability assessment, utilizing advanced Computer Vision techniques to address the increasing challenges of waste management. The project aims to contribute to a sustainable environment by promoting proper waste disposal and recycling practices.

This README.md provides a concise overview of the Automated Garbage Image Classifier project, highlighting its features, dataset, model architecture, and experimental analysis.
