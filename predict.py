import argparse
import os
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import tensorflow as tf

import config


def load_trained_model(model_path: str = config.MODEL_SAVE_PATH) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[Error] Model not found at '{model_path}'.\n"
            f"   Run 'python train.py' first to train and save the model."
        )

    model = tf.keras.models.load_model(model_path)
    print(f"[OK] Model loaded from: {model_path}")
    return model


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[Error] Image not found: '{image_path}'")

    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)

    original = img_array.astype(np.uint8)

    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch, original


def predict_gesture(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    class_names: list[str],
) -> tuple[str, float]:
    predictions = model.predict(img_batch, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100

    predicted_class = class_names[predicted_index]
    return predicted_class, confidence


def display_prediction(
    original_image: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title(
        f"Predicted: {predicted_class}  ({confidence:.1f}%)",
        fontsize=14,
        fontweight="bold",
        color="green" if confidence > 80 else "orange",
    )
    plt.tight_layout()
    plt.show()


def get_class_names() -> list[str]:
    class_names = sorted(
        entry
        for entry in os.listdir(config.DATASET_DIR)
        if os.path.isdir(os.path.join(config.DATASET_DIR, entry))
    )
    return class_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the hand gesture in a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input hand gesture image.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Hand Gesture Recognition - Single-Image Prediction")
    print("=" * 60)

    model = load_trained_model()

    class_names = get_class_names()
    print(f"   Classes: {class_names}\n")

    img_batch, original = preprocess_image(args.image)

    predicted_class, confidence = predict_gesture(model, img_batch, class_names)
    print(f"\nPrediction:  {predicted_class}")
    print(f"   Confidence:  {confidence:.1f}%\n")

    display_prediction(original, predicted_class, confidence)


if __name__ == "__main__":
    main()
