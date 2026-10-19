"""Central place for paths and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "cat_dog_model.keras"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

# Folder names are alphabetical -> label 0 = cats, label 1 = dogs
CLASS_FOLDERS = ["cats", "dogs"]
CLASS_NAMES = ["cat", "dog"]

HEAD_EPOCHS = 5        # stage 1: train only the new classifier head
FINE_TUNE_EPOCHS = 5   # stage 2: fine-tune top layers of MobileNetV2
FINE_TUNE_LAYERS = 30
