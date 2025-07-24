import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# Download the model from Google Drive using gdown
model_path = "efficientnetb0_model.keras"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/uc?id=1oDoZko1JrCbUxr0MYe9x1-FtoeyLkEqJ"  # Replace with your File ID
        gdown.download(url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title(" Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI image and get the tumor prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array / 255.0, axis=0)  # Normalize

    # Prediction
    prediction = model.predict(image_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"### 🧪 Predicted: `{pred_class}`")
    st.info(f"Confidence: `{confidence * 100:.2f}%`")
