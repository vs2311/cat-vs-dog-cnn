"""Predict a single image.   Usage: python -m src.predict path/to/image.jpg"""
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

from . import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(config.MODEL_PATH)
    return _model


def predict_proba(pil_image):
    """Return P(dog) for a PIL image."""
    img = pil_image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)
    return float(get_model().predict(arr, verbose=0)[0][0])


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    p = predict_proba(Image.open(sys.argv[1]))
    label = config.CLASS_NAMES[int(p > 0.5)]
    conf = p if p > 0.5 else 1 - p
    print(f"Prediction: {label} ({conf:.1%} confident)")


if __name__ == "__main__":
    main()
