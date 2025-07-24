import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model if not already present
model_url = "https://drive.google.com/uc?id=1oDoZko1JrCbUxr0MYe9x1-FtoeyLkEqJ"
model_path = "efficientnetb0_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify tumor type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Predicted Tumor Type: **{predicted_class}**")
