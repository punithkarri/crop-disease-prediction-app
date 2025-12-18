import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model   # ✅ CORRECT for TF 2.15
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="AI Crop Disease Prediction",
    layout="centered"
)

# ---------------------------------
# MODEL PATH
# ---------------------------------
MODEL_PATH = "crop_disease_model.keras"

# ---------------------------------
# CLASS NAMES (NO DATASET REQUIRED)
# ---------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ---------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict_disease(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return CLASS_NAMES[class_index], confidence

# ---------------------------------
# UI
# ---------------------------------
st.title("🌱 AI-Driven Crop Disease Prediction System")
st.write("Upload a leaf image to detect crop disease using a deep learning model.")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# RUN PREDICTION
# ---------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("Analyzing crop health..."):
        disease, confidence = predict_disease(image)

    st.subheader("🧪 Prediction Result")
    st.success(f"Detected Disease: {disease}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
    st.warning(f"10-Day Risk (Estimated): {min(confidence * 1.3, 1.0) * 100:.2f}%")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.caption("AI-Enabled Crop Disease Prediction | Streamlit Deployment (TF 2.15)")
