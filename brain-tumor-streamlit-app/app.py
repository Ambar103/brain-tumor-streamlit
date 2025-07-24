import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ------------------------------
# Title
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify the type of brain tumor using EfficientNetB0 model.")

# ------------------------------
# Download model from Google Drive
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=1aIXL1fPUUGoA4r9eBhd6ORuh1HVcYqpd"  # Replace with your file ID
    model_path = "efficientnetb0_model.h5"
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ------------------------------
# Label map
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ------------------------------
# Image upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.markdown("### 🧾 Prediction:")
    st.success(f"**{predicted_class}**")
