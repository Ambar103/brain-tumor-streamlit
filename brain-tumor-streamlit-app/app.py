import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set up title and description
st.title("Brain Tumor MRI Classification")
st.write("Upload a brain MRI scan image and classify tumor type using EfficientNetB0 model.")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("efficientnetb0_model.keras")
    return model

model = load_model()

# Define class names (edit if yours are different)
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Image upload
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.subheader("Prediction")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")
