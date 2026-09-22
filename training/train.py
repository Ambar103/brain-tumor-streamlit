"""Train an EfficientNetB0 brain tumor MRI classifier.

Downloads the public Kaggle "Brain Tumor MRI Dataset" (glioma, meningioma,
notumor, pituitary), fine-tunes EfficientNetB0 on it, evaluates on the held
-out test split, and saves the trained model + class names for the app.
"""
import json
import os
import shutil
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "efficientnetb0_model.keras"


def get_dataset_dirs():
    dataset_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    train_dir = dataset_path / "Training"
    test_dir = dataset_path / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        # some dataset versions nest an extra folder level
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1, f"Unexpected dataset layout: {list(dataset_path.iterdir())}"
        train_dir = subdirs[0] / "Training"
        test_dir = subdirs[0] / "Testing"
    return train_dir, test_dir


def build_datasets(train_dir, test_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )
    class_names = train_ds.class_names

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    def prep_train(x, y):
        x = augment(x)
        return preprocess(x), y

    def prep_eval(x, y):
        return preprocess(x), y

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(prep_train, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    train_dir, test_dir = get_dataset_dirs()
    print(f"Dataset: train={train_dir} test={test_dir}")
    train_ds, val_ds, test_ds, class_names = build_datasets(train_dir, test_dir)
    print("Classes:", class_names)

    model, base = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = ARTIFACTS_DIR / "checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("=== Phase 1: training head (frozen base) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

    print("=== Phase 2: fine-tuning top layers ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    print("=== Evaluating on test set ===")
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    (ARTIFACTS_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Brain Tumor MRI Classifier")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png")

    (ARTIFACTS_DIR / "class_names.json").write_text(json.dumps(class_names))

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Done. Artifacts in", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
