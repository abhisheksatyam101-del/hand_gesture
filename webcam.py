import os
import numpy as np
import cv2
import tensorflow as tf

import config

# ---------------------------------------------------------------------------
# TFLite path (fast CPU inference on Windows)
# ---------------------------------------------------------------------------
TFLITE_PATH = os.path.join("models", "hand_gesture_finetuned.tflite")


def load_tflite_model(tflite_path: str):
    """Load a TFLite FlatBuffer and return (interpreter, input_details, output_details)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[OK] TFLite model loaded from: {tflite_path}")
    return interpreter, input_details, output_details


def predict_tflite(interpreter, input_details, output_details, img_batch: np.ndarray) -> np.ndarray:
    """Run a single inference through the TFLite interpreter."""
    interpreter.set_tensor(input_details[0]["index"], img_batch)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


# ---------------------------------------------------------------------------
# Full-Keras path (fallback)
# ---------------------------------------------------------------------------

def load_trained_model(model_path: str = config.FINETUNE_SAVE_PATH) -> tf.keras.Model:
    if not os.path.exists(model_path):
        fallback_path = config.MODEL_SAVE_PATH
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                f"[Error] Neither fine-tuned model '{model_path}' nor base model '{fallback_path}' was found.\n"
                f"   Run 'python train.py' and 'python finetune.py' first."
            )
        print(f"[Warning] Fine-tuned model not found at '{model_path}'.\n"
              f"          Falling back to base model: '{fallback_path}'")
        model_path = fallback_path

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"[OK] Model loaded from: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(rgb_frame, config.IMG_SIZE)
    img_array = resized.astype(np.float32)           # TFLite expects float32
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def get_class_names() -> list[str]:
    class_names = sorted(
        entry
        for entry in os.listdir(config.DATASET_DIR)
        if os.path.isdir(os.path.join(config.DATASET_DIR, entry))
    )
    return class_names


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_webcam(predict_fn, class_names: list[str]) -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam. Please check your camera connection.")
        return

    print("\nWebcam started! Press 'q' to quit.\n")

    box_size = 250

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        start_x = (w - box_size) // 2
        start_y = (h - box_size) // 2
        end_x   = start_x + box_size
        end_y   = start_y + box_size

        roi       = frame[start_y:end_y, start_x:end_x]
        img_batch = preprocess_frame(roi)

        predictions     = predict_fn(img_batch)
        predicted_index = int(np.argmax(predictions))
        confidence      = float(predictions[predicted_index]) * 100
        predicted_class = class_names[predicted_index]

        label_text = f"{predicted_class}: {confidence:.1f}%"
        colour     = (0, 255, 0) if confidence > 80 else (0, 165, 255)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colour, 2)
        cv2.putText(
            img=frame, text="Place Hand Here",
            org=(start_x, start_y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
            color=colour, thickness=2, lineType=cv2.LINE_AA,
        )

        cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
        cv2.putText(
            img=frame, text=label_text, org=(20, 45),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
            color=colour, thickness=2, lineType=cv2.LINE_AA,
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed.\n")


def main() -> None:
    print("=" * 60)
    print("  Hand Gesture Recognition - Real-Time Webcam")
    print("=" * 60)

    class_names = get_class_names()
    print(f"   Classes: {class_names}\n")

    if os.path.exists(TFLITE_PATH):
        # ── Fast path: TFLite (loads in <1s, no GPU needed) ──────────────
        print(f"[INFO] Using TFLite model for fast CPU inference.")
        interpreter, input_details, output_details = load_tflite_model(TFLITE_PATH)
        predict_fn = lambda img: predict_tflite(interpreter, input_details, output_details, img)
    else:
        # ── Fallback: full Keras model (slow on CPU) ──────────────────────
        print(f"[INFO] TFLite model not found at '{TFLITE_PATH}'.")
        print(f"       Tip: run 'wsl bash run_in_wsl.sh convert_to_tflite.py' for faster inference.")
        model = load_trained_model()
        predict_fn = lambda img: model.predict(img, verbose=0)[0]

    run_webcam(predict_fn, class_names)


if __name__ == "__main__":
    main()
