import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

import numpy as np

def load_model():
    return keras.models.load_model('models/model_fruits.h5')

model = load_model()
classs = ['Banana', 'Orange', 'Apple']

st.title('Fruit Prediction App')
st.write('Upload an image of a fruit and the app will predict whether it is a Banana, Orange, or Apple.')


uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )