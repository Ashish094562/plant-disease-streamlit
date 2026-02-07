import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REPO_ID = "Ashish094562/plant-model-float32-tflite"

MODEL_FILE = "plant_model_float32.tflite"
DISEASE_JSON = "plant_disease.json"
LABELS_JSON = "class_names.json"

IMAGE_SIZE = 160

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# --------------------------------------------------
# DOWNLOAD FILES FROM HF
# --------------------------------------------------
@st.cache_resource
def download_assets():
    model_path = hf_hub_download(REPO_ID, MODEL_FILE)
    disease_path = hf_hub_download(REPO_ID, DISEASE_JSON)
    labels_path = hf_hub_download(REPO_ID, LABELS_JSON)
    return model_path, disease_path, labels_path

MODEL_PATH, DISEASE_PATH, LABELS_PATH = download_assets()

# --------------------------------------------------
# LOAD TFLITE MODEL
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
# LOAD LABELS
# --------------------------------------------------
with open(LABELS_PATH, "r") as f:
    IDX_TO_LABEL = {int(k): v for k, v in json.load(f).items()}

# --------------------------------------------------
# LOAD DISEASE INFO
# --------------------------------------------------
with open(DISEASE_PATH, "r", encoding="utf-8") as f:
    disease_list = json.load(f)

DISEASE_INFO = {d["name"]: d for d in disease_list}

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(image)

    if input_details[0]["dtype"] == np.float32:
        img = img.astype(np.float32)
    else:
        img = img.astype(np.uint8)

    img = np.expand_dims(img, axis=0)
    return img

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict(image: Image.Image):
    x = preprocess(image)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    label = IDX_TO_LABEL[top_idx]
    info = DISEASE_INFO[label]

    top3 = np.argsort(probs)[-3:][::-1]

    return label, confidence, info, probs, top3

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using a **TFLite model**.")

uploaded_file = st.file_uploader(
    "📷 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("🔍 Analyzing image..."):
        label, confidence, info, probs, top3 = predict(image)

    st.success("✅ Prediction Complete")

    st.subheader("🌿 Prediction")
    st.write(f"**Disease:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.subheader("🦠 Cause")
    st.write(info["cause"])

    st.subheader("💊 Recommended Treatment")
    st.write(info["cure"])

    st.subheader("📊 Top Predictions")
    for idx in top3:
        st.write(
            f"{IDX_TO_LABEL[int(idx)]} → {probs[int(idx)] * 100:.2f}%"
        )
