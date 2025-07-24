import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Load model from Google Drive (only once)
@st.cache_resource
def load_model():
    model_path = "efficientnetb0_model.h5"
    if not os.path.exists(model_path):
        file_id = "1aIXL1fPUUGoA4r9eBhd6ORuh1HVcYqpd"  # Replace this
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# UI
st.title("Brain Tumor MRI Classifier")
st.write("Upload an MRI image to predict tumor class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Loading model and predicting..."):
        model = load_model()
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]  # Change to your actual class names
    st.success(f"Predicted: {classes[predicted_class]} ({confidence:.2f} confidence)")
