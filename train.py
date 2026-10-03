import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config

# Reduce TensorFlow thread usage to lower memory pressure
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
# Ensure GPU memory growth (no effect on CPU-only but safe)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def load_data() -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.DATASET_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.DATASET_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
    )

    class_names: list[str] = train_ds.class_names
    print(f"\n[OK] Dataset loaded successfully!")
    print(f"   Classes ({len(class_names)}): {class_names}")
    print(f"   Training batches:   {len(train_ds)}")
    print(f"   Validation batches: {len(val_ds)}\n")

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_augmentation_layer() -> tf.keras.Sequential:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.15),
            tf.keras.layers.RandomZoom(height_factor=0.15),
            tf.keras.layers.RandomContrast(factor=0.2),
            tf.keras.layers.RandomTranslation(height_factor=0.15, width_factor=0.15, fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(factor=0.2),
        ],
        name="data_augmentation",
    )
    return augmentation


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3), name="input_image")

    x = build_augmentation_layer()(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout_1")(x)
    x = tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="HandGesture_MobileNetV2")

    model.summary()

    return model


def get_callbacks() -> list[tf.keras.callbacks.Callback]:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> tf.keras.callbacks.History:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nStarting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(),
    )

    print(f"\n[OK] Training complete!  Model saved to: {config.MODEL_SAVE_PATH}\n")
    return history


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def main() -> None:
    print("=" * 60)
    print("  Hand Gesture Recognition - Training Pipeline")
    print("=" * 60)

    train_ds, val_ds, class_names = load_data()

    model = build_model()

    history = train_model(model, train_ds, val_ds)

    plot_training_history(history)

    evaluate_model(model, val_ds, class_names)

    print("\n[OK] All done! Check the outputs/ folder for plots.\n")


if __name__ == "__main__":
    main()
