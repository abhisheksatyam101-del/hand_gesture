import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import config
from train import load_data

# Ensure GPU memory growth
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def evaluate(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    print(f"Loading {model_path}...")
    model = tf.keras.models.load_model(model_path)
    train_ds, val_ds, class_names = load_data()
    
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy for {model_path}: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return acc

if __name__ == "__main__":
    print("Evaluating Base Model:")
    evaluate(config.MODEL_SAVE_PATH)
    print("\nEvaluating Fine-tuned Model:")
    evaluate(config.FINETUNE_SAVE_PATH)
