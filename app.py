import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

REPO_ID = "Ashish094562/plant-disease-tflite"
MODEL_FILE = "plant_model_quant.tflite"
JSON_FILE = "plant_disease.json"
IMAGE_SIZE = 160

# --------------------------------------------------
# DOWNLOAD MODEL + JSON FROM HUGGING FACE
# --------------------------------------------------
@st.cache_resource
def download_assets():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    json_path = hf_hub_download(repo_id=REPO_ID, filename=JSON_FILE)
    return model_path, json_path

MODEL_PATH, JSON_PATH = download_assets()

# --------------------------------------------------
# LOAD TFLITE MODEL (USING TENSORFLOW)
# --------------------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------
LABELS = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Background_without_leaves','Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight',
    'Corn___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# --------------------------------------------------
# LOAD DISEASE INFO
# --------------------------------------------------
with open(JSON_PATH, "r") as f:
    disease_list = json.load(f)

DISEASE_INFO = {d["name"]: d for d in disease_list}

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_disease(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    idx = int(np.argmax(output))
    confidence = float(np.max(output))
    return LABELS[idx], confidence

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using a deep learning model.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        label, confidence = predict_disease(image)

    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    if label in DISEASE_INFO:
        st.subheader("🦠 Cause")
        st.write(DISEASE_INFO[label]["cause"])

        st.subheader("💊 Cure / Prevention")
        st.write(DISEASE_INFO[label]["cure"])
