"""MobileNetV2 transfer-learning model."""
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMG_SIZE


def build_model():
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze pretrained weights first

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)  # only active during training
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(dog)
    return keras.Model(inputs, outputs), base
