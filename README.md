# 📈 Multi-Label Classification of Thoracic Diseases using Chest X-Rays

This repository provides a complete pipeline to detect multiple thoracic diseases from chest X-ray images using both deep learning (CNNs) and classical machine learning (Random Forest, XGBoost) techniques. The main model leverages *DenseNet121, initialized with **ImageNet* or *CheXNet* pretrained weights depending on the pipeline stage.

---

## 📁 Repository Overview

| File | Description |
|------|-------------|
| ⁠ Custom_CNN.ipynb ⁠ | *Main deep learning pipeline*. Uses DenseNet121 (ImageNet weights) to train on X-ray data using Focal Loss. Saves trained weights, evaluation plots, and metrics. |
| ⁠ Feature_extractor.ipynb ⁠ | Extracts features using CheXNet-pretrained DenseNet121. Outputs a 1025-column CSV: 1 column for image ID and 1024 for features. |
| ⁠ processed_data.ipynb ⁠ | Handles augmentation to address label imbalance using *MLSMOTE* and *centroid-based augmentation*. |
| ⁠ RandomForest.ipynb ⁠ | Trains a Random Forest classifier on the augmented features. |
| ⁠ XGBoost.ipynb ⁠ | Trains an XGBoost classifier on the same augmented features. |

---

## 🔄 Workflow Summary

### Step 1: Deep Learning Model
•⁠  ⁠Uses DenseNet121 with ImageNet weights
•⁠  ⁠Trained using ⁠ train_data.csv ⁠ and ⁠ IMAGE_FOLDER ⁠
•⁠  ⁠Optimized using Focal Loss for class imbalance
•⁠  ⁠Automatically tunes thresholds per label based on F1-score
•⁠  ⁠Saves:
  - Trained model weights (⁠ chexnet_finetuned_weights.pth ⁠)
  - Evaluation metrics (AUC, Precision, Recall, F1)
  - ROC & PR Curves (PNG)

### Step 2: Feature Extraction
•⁠  ⁠Input: Chest X-ray images (NIH-14 dataset)
•⁠  ⁠Model: DenseNet121 + CheXNet weights (⁠ CheXNet_Keras_0.3.0_weights.h5 ⁠)
•⁠  ⁠Output: ⁠ extracted_features.csv ⁠

### Step 3: Data Augmentation
•⁠  ⁠Input: ⁠ extracted_features.csv ⁠ + ⁠ custom_labels.csv ⁠
•⁠  ⁠Output: ⁠ nih_chestxray_augmented.csv ⁠
•⁠  ⁠Techniques:
  - MLSMOTE
  - Centroid-based augmentation

### Step 4: Classical ML Models
•⁠  ⁠Models: Random Forest, XGBoost
•⁠  ⁠Input: ⁠ nih_chestxray_augmented.csv ⁠
•⁠  ⁠Evaluation done per disease label

---

## 🔍 Required Files

All necessary files are available here:
[Required_data_files](https://drive.google.com/drive/folders/1dzG3DnoCXYDNkSfDaNx1_73T-h-2tvxl?usp=drive_link)

| File | Purpose |
|------|---------|
| ⁠ train_data.csv ⁠ | Image IDs and labels for ⁠ Custom_CNN.ipynb ⁠ training |
| ⁠ nih_chestxray_augmented.csv ⁠ | Output of ⁠ processed_data.ipynb ⁠, input for ⁠ RandomForest.ipynb ⁠ & ⁠ XGBoost.ipynb ⁠ |
| ⁠ extracted_features.csv ⁠ | Output of ⁠ Feature_extractor.ipynb ⁠, input for augmentation |
| ⁠ custom_labels.csv ⁠ | One-hot encoded labels used during augmentation |
| ⁠ CheXNet_Keras_0.3.0_weights.h5 ⁠ | Pretrained weights for CheXNet model |

*Note*: For feature extraction, download NIH-14 Chest X-ray dataset from:
[Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

The X-ray ⁠ IMAGE_FOLDER ⁠ used in ⁠ Custom_CNN.ipynb ⁠ is also shared via Google Drive.
[Image_folder_custom_CNN](https://drive.google.com/drive/folders/1KBTK6euLKv9hJ7M4TfhSwQ8G0043zFsM?usp=sharing)

---

## 📊 Metrics & Evaluation

All models are evaluated using:
•⁠  ⁠Macro AUC
•⁠  ⁠Precision / Recall / F1 Score (macro and per class)
•⁠  ⁠ROC & PR Curve plots
•⁠  ⁠Threshold tuning per label for optimal F1

---

## 📅 Installation

⁠ bash
pip install torch torchvision scikit-learn xgboost imbalanced-learn pandas numpy matplotlib tqdm opencv-python
 ⁠

---

## 🙌 Contributors

This repository was created as part of an academic project focused on medical image understanding using deep and traditional ML approaches.

---

## 📚 License

This code is intended for academic and research use only. Ensure that NIH ChestX-ray14 dataset usage complies with NIH terms.
