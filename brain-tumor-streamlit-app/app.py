import json

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from gradcam import make_gradcam_heatmap, overlay_heatmap

HF_REPO_ID = "Ambar10/brain-tumor-efficientnetb0"
LOW_CONFIDENCE_THRESHOLD = 0.6


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="efficientnetb0_model.keras")
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource
def load_class_names():
    class_names_path = hf_hub_download(repo_id=HF_REPO_ID, filename="class_names.json")
    with open(class_names_path) as f:
        return json.load(f)


def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)


st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")

with st.sidebar:
    st.header("About")
    st.write(
        "EfficientNetB0 fine-tuned on the Kaggle Brain Tumor MRI Dataset "
        "(glioma, meningioma, pituitary, no tumor). Test accuracy: ~82.5%."
    )
    st.write("Grad-CAM highlights the MRI region that most influenced the prediction.")
    st.markdown("[View source on GitHub](https://github.com/Ambar103/brain-tumor-streamlit)")

st.title("Brain Tumor MRI Classifier")
st.write("Upload an MRI image to predict tumor class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image. Please upload a JPG or PNG MRI scan.")
        st.stop()

    with st.spinner("Predicting..."):
        model = load_model()
        class_names = load_class_names()
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        heatmap = make_gradcam_heatmap(processed, model, pred_index=predicted_index)
        overlay = overlay_heatmap(image, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM: regions driving the prediction", use_container_width=True)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(f"Prediction: {class_names[predicted_index]} (Confidence: {confidence:.2%}) — low confidence, treat with caution.")
    else:
        st.success(f"Prediction: {class_names[predicted_index]} (Confidence: {confidence:.2%})")

    st.bar_chart({name: float(prob) for name, prob in zip(class_names, predictions)})

    st.caption("This tool is for educational/demo purposes only and is not a medical diagnosis.")
