"""
finetune.py
-----------
Fine-tunes the already-trained hand gesture model by unfreezing the top
layers of the MobileNetV2 backbone and retraining at a very low learning
rate. This avoids training from scratch and typically lifts accuracy by
5-10 percentage points.
"""

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
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


# ---------------------------------------------------------------------------
# Data loading (identical to train.py)
# ---------------------------------------------------------------------------

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
    print(f"\n[OK] Dataset loaded — {len(class_names)} classes, "
          f"{len(train_ds)} train batches, {len(val_ds)} val batches\n")
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------------
# Fine-tune setup
# ---------------------------------------------------------------------------

def rebuild_model_from_weights(weights_path: str) -> tf.keras.Model:
    """
    Rebuilds the MobileNetV2 model architecture from scratch and loads
    weights from a .h5 file. Used for TF 2.10 compatibility when the
    .keras format (Keras 3) cannot be loaded directly.
    """
    print(f"[INFO] Rebuilding model architecture and loading weights from: {weights_path}")

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3), name="input_image")
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
    x = augmentation(inputs)
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

    if weights_path.endswith(".weights.h5"):
        model.load_weights(weights_path, skip_mismatch=True)
    else:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print(f"[OK] Weights loaded successfully.")
    return model


def prepare_for_finetuning(model: tf.keras.Model) -> tf.keras.Model:
    """
    Finds the MobileNetV2 base inside the model, freezes all its layers,
    then unfreezes only the last FINETUNE_UNFREEZE_LAYERS layers.
    """
    # Locate the MobileNetV2 sub-model.
    # We pick the nested Model with the MOST layers to avoid matching the
    # small data_augmentation Sequential (6 layers) instead of MobileNetV2 (154 layers).
    base_model: tf.keras.Model | None = None
    max_layers = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > max_layers:
            base_model = layer
            max_layers = len(layer.layers)

    if base_model is None:
        raise RuntimeError("Could not find a nested Keras model (MobileNetV2) inside the loaded model.")

    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - config.FINETUNE_UNFREEZE_LAYERS)

    print(f"[INFO] MobileNetV2 has {total_layers} layers.")
    print(f"[INFO] Freezing layers 0–{freeze_until - 1}, unfreezing layers {freeze_until}–{total_layers - 1}.")

    base_model.trainable = True
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Recompile at a much lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable_params = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    print(f"[INFO] Trainable parameters after unfreezing: {trainable_params:,}\n")
    return model


def get_finetune_callbacks() -> list[tf.keras.callbacks.Callback]:
    os.makedirs("models", exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.FINETUNE_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation & plotting (same as train.py)
# ---------------------------------------------------------------------------

def plot_finetune_history(history: tf.keras.callbacks.History) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2)
    axes[0].set_title("Fine-tune Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Fine-tune Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "finetune_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Fine-tune curves saved to: {save_path}")


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    cm_filename: str = "confusion_matrix_finetuned.png",
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
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Fine-tuned)", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, cm_filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Hand Gesture Recognition - Fine-Tuning Pipeline")
    print("=" * 60)

    # 1. Load data
    train_ds, val_ds, class_names = load_data()

    # 2. Load the best model from Phase 1.
    # Prefer the full .keras checkpoint (saved by train.py) over the legacy
    # model.weights.h5 extracted from an older run.  The .weights.h5 path is
    # kept only as a final fallback so the file isn't broken for anyone who
    # genuinely only has that artifact.
    weights_h5 = os.path.join("models", "model.weights.h5")
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[INFO] Loading model from: {config.MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    elif os.path.exists(weights_h5):
        # Legacy fallback: rebuild architecture and load extracted weights
        model = rebuild_model_from_weights(weights_h5)
    else:
        raise FileNotFoundError(
            f"No trained model found at '{config.MODEL_SAVE_PATH}' or '{weights_h5}'.\n"
            "Run 'python train.py' first."
        )

    # 3. Unfreeze top layers and recompile at low LR
    model = prepare_for_finetuning(model)

    # 4. Fine-tune
    print(f"\nStarting fine-tuning for up to {config.FINETUNE_EPOCHS} epochs "
          f"(LR={config.FINETUNE_LR}, unfreezing top {config.FINETUNE_UNFREEZE_LAYERS} layers)...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINETUNE_EPOCHS,
        callbacks=get_finetune_callbacks(),
    )

    print(f"\n[OK] Fine-tuning complete! Model saved to: {config.FINETUNE_SAVE_PATH}\n")

    # 5. Plots & evaluation
    plot_finetune_history(history)
    evaluate_model(model, val_ds, class_names)

    print("\n[OK] All done! Check the outputs/ folder for plots.\n")


if __name__ == "__main__":
    main()
