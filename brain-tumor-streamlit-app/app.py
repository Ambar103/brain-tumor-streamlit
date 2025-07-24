import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

@st.cache_resource
def load_model():
    model_path = "efficientnetb0_model.keras"
    if not os.path.exists(model_path):
        file_id = 1JqarGvmPU4-r6Vub5QlWfCg1a9VFqsVB  # Replace this
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

st.title("Brain Tumor MRI Classifier")
st.write("Upload an MRI image to predict tumor class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        model = load_model()
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    st.success(f"Prediction: {classes[predicted_class]} (Confidence: {confidence:.2f})")
