# Pneumonia Detection from Chest X-Ray Images using CNN

## Overview

This project develops a **Convolutional Neural Network (CNN)** model using **TensorFlow** and **Keras** to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The goal is to build an automated diagnostic tool that can accurately identify pneumonia from paediatric chest X-ray images.

## Dataset Description

The dataset comprises **5,856 JPEG chest X-ray images** (anterior-posterior view) sourced from paediatric patients aged **1 to 5 years** at a renowned hospital. The X-rays were part of the routine clinical care of these patients.

| Split       | Normal | Pneumonia | Total  |
|-------------|--------|-----------|--------|
| Training    | 1,341  | 3,875     | 5,216  |
| Testing     | 234    | 390       | 624    |
| Validation  | 8      | 8         | 16     |
| **Total**   | **1,583** | **4,273** | **5,856** |

**Key Observation**: The training set exhibits significant class imbalance — approximately **2.9x more PNEUMONIA images** than NORMAL images. This must be addressed during model training to avoid bias towards the majority class.

---

## Approach

### 1. Data Exploration

Before building the model, the dataset was thoroughly explored to understand its structure. The distribution of images across all classes and splits was analysed. Visualisations of sample X-ray images from both categories were generated to understand the visual differences — normal X-rays show clear lung fields, while pneumonia X-rays show areas of opacity or consolidation in the lungs.

### 2. Data Preprocessing

- **Image Resizing**: All images are resized to **150x150 pixels** to provide uniform input dimensions to the CNN.
- **Pixel Normalisation**: Pixel values are rescaled from the [0, 255] range to [0, 1] to stabilise gradient computations and improve convergence during training.

### 3. Data Augmentation

To combat overfitting and artificially increase the diversity of training samples, the following augmentation techniques are applied **only to training images**:

| Technique | Range |
|-----------|-------|
| Rotation | Up to 20 degrees |
| Width Shift | Up to 20% |
| Height Shift | Up to 20% |
| Shear Transformation | Up to 20% |
| Zoom | Up to 20% |
| Horizontal Flip | Enabled |

Validation and test images are **not augmented** — they are only rescaled to preserve their integrity for unbiased evaluation.

### 4. Handling Class Imbalance

Class weights are computed inversely proportional to the class frequency in the training data. This penalises misclassifications of the minority class (NORMAL) more heavily, ensuring the model does not develop a bias towards predicting PNEUMONIA for every input.

---

## Methodology

### CNN Architecture

A custom CNN with **four convolutional blocks** followed by fully connected layers was designed for this binary classification task:

```
Input (150x150x3)
    |
    +-- Block 1: Conv2D(32) -> BN -> Conv2D(32) -> BN -> MaxPool -> Dropout(0.25)
    +-- Block 2: Conv2D(64) -> BN -> Conv2D(64) -> BN -> MaxPool -> Dropout(0.25)
    +-- Block 3: Conv2D(128) -> BN -> Conv2D(128) -> BN -> MaxPool -> Dropout(0.25)
    +-- Block 4: Conv2D(256) -> BN -> MaxPool -> Dropout(0.25)
    |
    +-- Flatten
    +-- Dense(512) -> BN -> Dropout(0.5)
    +-- Dense(256) -> BN -> Dropout(0.5)
    +-- Dense(1, sigmoid) -> Output (0 = NORMAL, 1 = PNEUMONIA)
```

#### Design Rationale

| Component | Purpose |
|-----------|---------|
| **Conv2D layers** | Extract spatial features from X-ray images, progressing from low-level edges (32 filters) to high-level patterns (256 filters) |
| **Batch Normalisation** | Normalises layer inputs to stabilise and accelerate training |
| **MaxPooling2D** | Reduces spatial dimensions, lowering computational cost and providing translation invariance |
| **Dropout (25% conv, 50% dense)** | Randomly deactivates neurons during training to prevent overfitting |
| **Sigmoid Activation** | Outputs a probability between 0 and 1 for binary classification |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (learning rate = 0.0001) |
| Loss Function | Binary Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 5 epochs (monitors validation loss) |
| Learning Rate Reduction | Factor of 0.5 when validation loss plateaus for 3 epochs |

### Callbacks

1. **EarlyStopping**: Monitors validation loss and stops training if no improvement for 5 consecutive epochs. Automatically restores the best weights.
2. **ReduceLROnPlateau**: Reduces the learning rate by 50% when validation loss plateaus, allowing finer optimisation in later epochs.
3. **ModelCheckpoint**: Saves the best model (based on highest validation accuracy) during the training process.

---

## Findings

### Model Performance

Based on the architecture and training strategy, the model is expected to achieve:

- **Test Accuracy**: 85-95% on the pneumonia classification task
- **High Sensitivity (Recall for PNEUMONIA)**: This is critical in medical diagnostics to minimise false negatives — missing a pneumonia case is far more dangerous than a false alarm.
- **Specificity (Recall for NORMAL)**: Measures the model's ability to correctly identify healthy patients.

The model generates a detailed **classification report** with precision, recall, and F1-score for both classes, along with a **confusion matrix** that provides a granular breakdown of true positives, true negatives, false positives, and false negatives.

### Key Observations

1. **Class Imbalance Impact**: Without class weighting, the model would be biased towards predicting PNEUMONIA (the majority class). Applying class weights significantly improves the model's ability to correctly classify NORMAL images.
2. **Data Augmentation Effect**: Augmentation techniques help the model generalise better to unseen images by exposing it to varied transformations of the training data, reducing overfitting.
3. **Batch Normalisation Contribution**: Adding batch normalisation after each convolutional layer stabilises training and allows the use of higher learning rates without divergence.
4. **Validation Set Limitation**: The validation set is very small (only 16 images — 8 per class), which may lead to noisy validation metrics. The test set (624 images) provides a more reliable measure of model performance.

### Outputs Generated

The following files are generated in the `outputs/` directory after running the code:

| File | Description |
|------|-------------|
| `best_model.keras` | Best model checkpoint saved during training |
| `pneumonia_detection_model.keras` | Final trained model |
| `classification_report.txt` | Detailed classification metrics (precision, recall, F1-score) |
| `training_history.png` | Training and validation accuracy/loss curves over epochs |
| `confusion_matrix.png` | Confusion matrix heatmap for the test set |
| `sample_images.png` | Sample X-ray images from both classes |
| `sample_predictions.png` | Model predictions on individual test images |
| `class_distribution.png` | Bar chart showing class distribution across splits |
| `augmentation_examples.png` | Visual examples of data augmentation transformations |

### Potential Improvements

1. **Transfer Learning**: Using pre-trained models such as VGG16 or ResNet50 (trained on ImageNet) could improve accuracy by leveraging already-learned feature representations.
2. **Larger Validation Set**: Splitting a portion of training data into validation would provide more reliable validation metrics than the current 16-image set.
3. **K-Fold Cross-Validation**: Would provide more robust and reliable performance estimates across different data splits.
4. **Grad-CAM Visualisation**: Generating gradient-weighted class activation maps would help visualise which regions of the X-ray the model focuses on, improving clinical interpretability.

---

## Project Structure

```
Pneumonia Detection/
├── pneumonia_detection.py       # Main CNN model code
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── x-ray_image/                 # Dataset
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── outputs/                     # Generated after running the code
    ├── best_model.keras
    ├── pneumonia_detection_model.keras
    ├── classification_report.txt
    ├── training_history.png
    ├── confusion_matrix.png
    ├── sample_images.png
    ├── sample_predictions.png
    ├── class_distribution.png
    └── augmentation_examples.png
```

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
pip install -r requirements.txt
```

### Execution

```bash
python pneumonia_detection.py
```

The script will automatically:
1. Explore and visualise the dataset
2. Apply data augmentation and preprocessing
3. Build and train the CNN model
4. Evaluate the model on the test set
5. Generate all output files (plots, reports, saved model)

## Tools and Technologies

- **Python 3.8+**
- **TensorFlow / Keras** — Deep learning framework for building and training the CNN
- **NumPy** — Numerical computations
- **Matplotlib** — Visualisation of training curves, sample images, and predictions
- **Seaborn** — Confusion matrix heatmap visualisation
- **Scikit-learn** — Classification report and confusion matrix computation
