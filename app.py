import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REPO_ID = "Ashish094562/plant-disease-tflite"
MODEL_FILE = "plant_model_quant.tflite"
JSON_FILE = "plant_disease.json"
IMAGE_SIZE = 160

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# --------------------------------------------------
# DOWNLOAD FILES FROM HUGGING FACE
# --------------------------------------------------
@st.cache_resource
def load_files():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE
    )
    json_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=JSON_FILE
    )
    return model_path, json_path

model_path, json_path = load_files()

# --------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------------------------------
# LOAD DISEASE INFO
# --------------------------------------------------
with open(json_path, "r") as f:
    disease_list = json.load(f)

DISEASE_INFO = {d["name"]: d for d in disease_list}
LABELS = list(DISEASE_INFO.keys())

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess(image: Image.Image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict(image: Image.Image):
    x = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(output))
    conf = float(np.max(output))
    label = LABELS[idx]
    info = DISEASE_INFO[label]

    return label, conf, info["cause"], info["cure"]

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using a lightweight TFLite model.")

uploaded_file = st.file_uploader(
    "📷 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        label, confidence, cause, cure = predict(image)

    st.success("✅ Prediction Complete")

    st.subheader("🌿 Prediction")
    st.write(f"**Disease:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.subheader("🦠 Cause")
    st.write(cause)

    st.subheader("💊 Cure / Prevention")
    st.write(cure)
