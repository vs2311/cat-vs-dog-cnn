"""Streamlit web demo for the cat vs dog classifier."""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cat_dog_model.keras"
IMG_SIZE = (160, 160)

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict_dog_probability(img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)
    return float(load_model().predict(arr, verbose=0)[0][0])


st.title("🐾 Cat vs Dog Classifier")
st.caption("A CNN (MobileNetV2 transfer learning) built with TensorFlow/Keras. "
           "Test accuracy: 96.7% on 1,000 unseen images.")

file = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    image = Image.open(file)
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        p_dog = predict_dog_probability(image)

    label = "Dog 🐶" if p_dog > 0.5 else "Cat 🐱"
    confidence = max(p_dog, 1 - p_dog)
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: **{confidence:.1%}**")
    st.progress(1 - p_dog, text=f"Cat: {1 - p_dog:.1%}")
    st.progress(p_dog, text=f"Dog: {p_dog:.1%}")
else:
    st.info("Upload an image to get a prediction.")
