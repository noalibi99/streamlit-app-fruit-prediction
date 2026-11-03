import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Landscape Classifier", page_icon="🏞️")

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

@st.cache_resource
def load_model():
    return keras.models.load_model("models/model_initial.keras", compile=False)

model = load_model()

st.title("🏞️ Landscape Classifier")
st.caption("Upload a landscape photo to classify it into one of six categories.")

with st.expander("ℹ️ About the model"):
    st.write("CNN trained on the Intel Image Classification dataset.")
    st.write("**Classes:** buildings, forest, glacier, mountain, sea, street.")
    st.write("**Input size:** 150 × 150 pixels, RGB.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", width=400)
    st.caption(f"Original size: {image.width} × {image.height} pixels")

    if st.button("🔍 Predict", type="primary"):
        resized = image.resize((150, 150), Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32)[None, ...]

        with st.spinner("Analyzing image..."):
            probabilities = model.predict(x, verbose=0)[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", CLASSES[index].capitalize())
        col2.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

        st.subheader("Confidence by class")
        scores = pd.DataFrame({
            "Class": [name.capitalize() for name in CLASSES],
            "Confidence (%)": probabilities * 100,
        }).set_index("Class")

        st.bar_chart(
            scores,
            y="Confidence (%)",
            color="#4F86F7",
        )

        st.caption(
            "Scores are model estimates. The model always selects one of "
            "these six classes, even for unrelated images."
        )
else:
    st.info("👆 Upload an image to get started.")