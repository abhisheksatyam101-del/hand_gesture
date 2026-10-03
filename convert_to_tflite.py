"""
convert_to_tflite.py
Converts the fine-tuned Keras model to a TFLite FlatBuffer for fast CPU inference.
Run this in WSL where TF has GPU support.
"""
import os
import tensorflow as tf
import config

# Ensure GPU memory growth
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

model_path = config.FINETUNE_SAVE_PATH
tflite_path = os.path.join("models", "hand_gesture_finetuned.tflite")

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded. Converting to TFLite...")

# Convert – use default float32 (no quantisation) for max accuracy
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: dynamic-range quantisation (smaller & faster, ~1% accuracy loss)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"\n[OK] TFLite model saved to: {tflite_path}  ({size_mb:.1f} MB)")
print("You can now run webcam.py on Windows — it will use the TFLite model automatically.")
