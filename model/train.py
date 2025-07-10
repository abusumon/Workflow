# model/train.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

img_size = 224
batch_size = 32

def get_data_generators(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    val = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    test = val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return train, val, test

def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def plot_history(history):
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.show()

def main():
    data_dir = "data"  # Update this path as needed
    train, val, test = get_data_generators(data_dir)
    model = build_model()
    print(model.summary())
    history = model.fit(train, validation_data=val, epochs=5)
    plot_history(history)
    # Fine-tuning
    fine_tune_epochs = 5
    history_fine = model.fit(train, validation_data=val, epochs=fine_tune_epochs)
    # Evaluate
    loss, acc = model.evaluate(test)
    print("Loss:", loss)
    print("Accuracy:", acc)
    # Confusion matrix
    pred_probs = model.predict(test)
    pred_labels = (pred_probs > 0.5).astype("int32").flatten()
    true_labels = test.classes
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(true_labels, pred_labels, target_names=["Normal", "Pneumonia"]))
    # Save model
    model.save("../app/chest_xray_model.keras")

if __name__ == "__main__":
    main()
