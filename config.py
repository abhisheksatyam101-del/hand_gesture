import os

DATASET_DIR: str = os.path.join("dataset", "HG14", "HG14-Hand Gesture")

MODEL_SAVE_PATH: str = os.path.join("models", "hand_gesture_mobilenetv2.keras")

IMG_SIZE: tuple[int, int] = (240, 240)

BATCH_SIZE: int = 32
EPOCHS: int = 30
LEARNING_RATE: float = 5e-4
VALIDATION_SPLIT: float = 0.2

NUM_CLASSES: int = 14
DENSE_UNITS: int = 256
DROPOUT_RATE: float = 0.3

EARLY_STOPPING_PATIENCE: int = 8
REDUCE_LR_PATIENCE: int = 4
REDUCE_LR_FACTOR: float = 0.5

# Fine-tuning settings (used by finetune.py)
FINETUNE_LR: float = 2e-5          # Low LR to prevent catastrophic forgetting
FINETUNE_EPOCHS: int = 30          # More epochs for adaptation
FINETUNE_UNFREEZE_LAYERS: int = 50 # Unfreeze top 50 MobileNetV2 layers (conservative)
FINETUNE_SAVE_PATH: str = os.path.join("models", "hand_gesture_finetuned.keras")

OUTPUT_DIR: str = "outputs"

SEED: int = 42

