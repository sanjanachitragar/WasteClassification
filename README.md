# Waste Classification using Transfer Learning 🌱♻️

## Overview
Waste management is a pressing issue, leading to environmental problems. This project aims to automate waste classification into two classes: Organic and Recyclable. Using a machine learning model and IoT, we can contribute to better waste management practices.

## Dataset
Utilized the [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) from Kaggle, containing 22,564 training images and 2,513 testing images of organic and recyclable objects.

## Objectives
- **Pre-processing and Image Augmentation:**
  - Utilized Keras' ImageGeneratorClass for data pre-processing and augmentation.

- **Transfer Learning Steps:**
  1. Obtained a pre-trained model (VGG-16).
  2. Created a base model and loaded pre-trained weights (ImageNet).
  3. Froze layers in the base model.
  4. Defined a new model on top of the base model's output.
  5. Trained the resulting model on the dataset.

- **Model Building:**
  - Implemented an end-to-end VGG-16-based transfer learning model for binary image classification.

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   numpy==1.21.2
datetime==4.3
matplotlib==3.4.3
seaborn==0.11.2
tensorflow==2.6.0
keras==2.6.0
scikit-learn==0.24.2
opendatasets==0.1.20
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data) and place it in the `data` directory.

4. Run the Jupyter notebook or Python script for training and testing the model.

## Results
- Achieved high accuracy in waste classification.
- Contributed towards sustainable waste management practices.

## Future Improvements
- Explore additional pre-trained models.
- Enhance model performance through fine-tuning.
- Integrate the model with an IoT system for real-time waste classification.

Let's make a positive impact on our environment. 🌍🌿♻️
