import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

import numpy as np

def load_model():
    return keras.models.load_model('models/model_fruits.h5')

model = load_model()
classes = ['Apple', 'Banana', 'Orange']

st.title('Fruit Prediction App')
st.write('Upload an image of a fruit and the app will predict whether it is a Banana, Orange, or Apple.')


uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    img = image.resize((32, 32))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.subheader('Prediction Results')
    st.success(f'Predicted Fruit: {predicted_class} \nConfidence: {confidence:.2f}')
    for i in range(len(classes)):
        st.write(f'{classes[i]}: {prediction[0][i]:.2f}')
