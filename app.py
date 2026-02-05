import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REPO_ID = "Ashish094562/plant-disease-tflite"
MODEL_FILENAME = "plant_model_quant.tflite"
JSON_FILENAME = "plant_disease.json"
IMAGE_SIZE = 160

# --------------------------------------------------
# DOWNLOAD FILES FROM HUGGING FACE (PUBLIC REPO)
# --------------------------------------------------
@st.cache_resource
def load_files():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    json_path = hf_hub_download(repo_id=REPO_ID, filename=JSON_FILENAME)
    return model_path, json_path

MODEL_PATH, DISEASE_JSON = load_files()

# --------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------
labels = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Background_without_leaves','Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight',
    'Corn___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight',
    'Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# --------------------------------------------------
# LOAD DISEASE JSON
# --------------------------------------------------
with open(DISEASE_JSON, "r") as f:
    disease_list = json.load(f)

disease_info = {
    item["name"]: {
        "cause": item["cause"],
        "cure": item["cure"]
    }
    for item in disease_list
}

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = int(np.argmax(output))
    confidence = float(np.max(output))
    return labels[class_index], confidence

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using AI.")

uploaded_file = st.file_uploader(
    "📷 Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        label, confidence = predict(image)

    st.success("✅ Prediction Complete")

    st.subheader("🌿 Predicted Disease")
    st.write(f"**{label}**")
    st.write(f"**Confidence:** `{confidence * 100:.2f}%`")

    if label in disease_info:
        st.subheader("💡 Disease Details")
        st.markdown("**🦠 Cause**")
        st.write(disease_info[label]["cause"])
        st.markdown("**💊 Cure / Prevention**")
        st.write(disease_info[label]["cure"])
