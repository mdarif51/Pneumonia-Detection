"""
Pneumonia Detection from Chest X-Ray Images using Convolutional Neural Networks
================================================================================
This script builds, trains, and evaluates a CNN model using TensorFlow and Keras
to classify chest X-ray images as either NORMAL or PNEUMONIA.

Dataset: 5,863 JPEG chest X-ray images (anterior-posterior view) from paediatric
patients aged 1-5 years, organised into train, test, and val directories.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x-ray_image')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
VAL_DIR = os.path.join(BASE_DIR, 'val')

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================

def explore_dataset():
    """Print dataset statistics and class distribution."""
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    splits = {'Train': TRAIN_DIR, 'Test': TEST_DIR, 'Validation': VAL_DIR}
    total_images = 0

    for split_name, split_path in splits.items():
        normal_count = len(os.listdir(os.path.join(split_path, 'NORMAL')))
        pneumonia_count = len(os.listdir(os.path.join(split_path, 'PNEUMONIA')))
        split_total = normal_count + pneumonia_count
        total_images += split_total

        print(f"\n{split_name} Set:")
        print(f"  Normal:    {normal_count} images")
        print(f"  Pneumonia: {pneumonia_count} images")
        print(f"  Total:     {split_total} images")
        print(f"  Class Ratio (Pneumonia/Normal): {pneumonia_count / normal_count:.2f}")

    print(f"\nTotal Images in Dataset: {total_images}")
    print("=" * 60)


def plot_sample_images():
    """Display sample images from both classes."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold')

    for idx, category in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(TRAIN_DIR, category)
        images = os.listdir(folder)[:5]
        for i, img_name in enumerate(images):
            img_path = os.path.join(folder, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            axes[idx, i].imshow(img, cmap='gray')
            axes[idx, i].set_title(category, fontsize=10)
            axes[idx, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample images saved to outputs/sample_images.png")


# ============================================================================
# 3. DATA PREPROCESSING AND AUGMENTATION
# ============================================================================

def create_data_generators():
    """
    Create data generators with augmentation for training and rescaling for
    validation/test sets. Data augmentation helps prevent overfitting and
    improves generalisation, especially given the class imbalance.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED,
        shuffle=True
    )

    print("Loading validation data...")
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED,
        shuffle=False
    )

    print("Loading test data...")
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED,
        shuffle=False
    )

    print(f"\nClass Indices: {train_generator.class_indices}")
    return train_generator, val_generator, test_generator


# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================

def build_cnn_model():
    """
    Build a CNN model with multiple convolutional blocks, batch normalisation,
    and dropout for regularisation. The architecture progressively extracts
    features from low-level edges to high-level patterns in X-ray images.
    """
    model = Sequential([
        # Block 1: Initial feature extraction
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2: Mid-level feature extraction
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3: High-level feature extraction
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 4: Deep feature extraction
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    return model


def compile_model(model):
    """
    Compile the model with Adam optimiser and binary crossentropy loss.
    Binary crossentropy is appropriate for two-class classification.
    """
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    print("=" * 60)
    model.summary()
    return model


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def get_class_weights(train_generator):
    """
    Compute class weights to handle the imbalance between NORMAL and PNEUMONIA
    classes. The training set has roughly 3x more PNEUMONIA images than NORMAL.
    """
    total = train_generator.samples
    class_counts = np.bincount(train_generator.classes)
    class_weight = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    print(f"\nClass Weights: {class_weight}")
    return class_weight


def train_model(model, train_generator, val_generator):
    """Train the model with callbacks for early stopping and learning rate reduction."""
    class_weight = get_class_weights(train_generator)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("TRAINING THE MODEL")
    print("=" * 60)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ============================================================================
# 6. VISUALISATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved to outputs/training_history.png")


# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================

def evaluate_model(model, test_generator):
    """Evaluate the model on the test set and display detailed metrics."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Generate predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(true_classes, predicted_classes,
                                   target_names=class_labels)
    print(report)

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Pneumonia Detection - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n\n")
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(true_classes, predicted_classes, class_labels)

    return test_loss, test_accuracy


def plot_confusion_matrix(true_classes, predicted_classes, class_labels):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved to outputs/confusion_matrix.png")


# ============================================================================
# 8. PREDICTION ON INDIVIDUAL IMAGES
# ============================================================================

def predict_single_image(model, image_path):
    """Load a single image and predict whether it shows NORMAL or PNEUMONIA."""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {label}\nConfidence: {confidence:.2%}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    return label, confidence


# ============================================================================
# 9. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Run the complete pneumonia detection pipeline."""
    print("\n" + "=" * 60)
    print("  PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
    print("  Using Convolutional Neural Networks (TensorFlow/Keras)")
    print("=" * 60)

    # Step 1: Explore the dataset
    explore_dataset()

    # Step 2: Visualise sample images
    plot_sample_images()

    # Step 3: Create data generators with augmentation
    train_gen, val_gen, test_gen = create_data_generators()

    # Step 4: Build and compile the CNN model
    model = build_cnn_model()
    model = compile_model(model)

    # Step 5: Train the model
    history = train_model(model, train_gen, val_gen)

    # Step 6: Plot training history
    plot_training_history(history)

    # Step 7: Evaluate on test set
    test_loss, test_accuracy = evaluate_model(model, test_gen)

    # Step 8: Save the final model
    model.save(os.path.join(OUTPUT_DIR, 'pneumonia_detection_model.keras'))
    print(f"\nFinal model saved to outputs/pneumonia_detection_model.keras")

    # Step 9: Demonstrate single image prediction
    sample_normal = os.path.join(TEST_DIR, 'NORMAL', os.listdir(os.path.join(TEST_DIR, 'NORMAL'))[0])
    sample_pneumonia = os.path.join(TEST_DIR, 'PNEUMONIA', os.listdir(os.path.join(TEST_DIR, 'PNEUMONIA'))[0])

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    print("\nPredicting a NORMAL X-ray:")
    predict_single_image(model, sample_normal)

    print("\nPredicting a PNEUMONIA X-ray:")
    predict_single_image(model, sample_pneumonia)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
