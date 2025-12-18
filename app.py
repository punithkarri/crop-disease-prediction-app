import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="AI Crop Disease Prediction", layout="centered")

MODEL_PATH = "crop_disease_model.keras"

CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot","Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight","Grape___healthy",
    "Orange___Haunglongbing","Peach___Bacterial_spot","Peach___healthy",
    "Pepper___Bacterial_spot","Pepper___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

def predict_disease(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    idx = np.argmax(pred)
    return CLASS_NAMES[idx], float(np.max(pred))

st.title("🌱 AI-Driven Crop Disease Prediction System")
st.write("Upload a leaf image to detect crop disease using AI")

uploaded_file = st.file_uploader("Upload leaf image", ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)
    disease, confidence = predict_disease(image)
    st.success(f"Detected Disease: {disease}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.warning(f"10-Day Risk: {min(confidence*1.3,1)*100:.2f}%")
