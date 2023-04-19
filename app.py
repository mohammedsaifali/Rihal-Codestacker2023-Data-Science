import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import tensorflow as tf


def display_images(shabby, predicted):
    col1, col2 = st.columns(2)
    col1.image(shabby, use_column_width=True, caption="Shabby Image")
    col2.image(predicted, use_column_width=True, caption="Predicted Clean Image")\

st.title("Document Image Denoising")
st.text("Codestacker Challenge 2023 Data Science")
st.text("Author: Mohd Saif ALi")
st.text("Email: catchsaifali@gmail.com")

#st.sidebar.title("Model Selection")
#model_choice = st.sidebar.selectbox("Select a model:", ("TensorFlow", "PyTorch"))

uploaded_file = st.file_uploader("Upload a shabby image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.convert("L")  # Convert to grayscale
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize and reshape the input image
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

        # Load the TensorFlow model
    best_model = tf.keras.models.load_model("model_denoise.h5")
    predicted_img = best_model.predict(img_input)

    predicted_img = predicted_img.reshape(256, 256)

    display_images(img_resized, predicted_img)
