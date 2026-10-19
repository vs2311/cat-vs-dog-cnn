"""tf.data pipelines for train / val / test."""
import tensorflow as tf
from tensorflow import keras

from .config import BATCH_SIZE, CLASS_FOLDERS, DATA_DIR, IMG_SIZE, SEED


def _load(split, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=CLASS_FOLDERS,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def load_train():
    return _load("train", shuffle=True)


def load_val():
    return _load("val", shuffle=False)


def load_test():
    return _load("test", shuffle=False)
