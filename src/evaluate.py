"""Evaluate on the held-out test set.   Usage: python -m src.evaluate"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow import keras

from . import config
from .data import load_test


def main():
    model = keras.models.load_model(config.MODEL_PATH)
    test_ds = load_test()

    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix (test set)")
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    plt.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved confusion matrix -> outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
