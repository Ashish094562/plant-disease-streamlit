import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from huggingface_hub import hf_hub_download

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease")

# -----------------------------
# Load TFLite model from Hugging Face
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Ashish094562/plant-disease-model",
        filename="model.tflite"
    )
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = [
    "Healthy",
    "Leaf Blight",
    "Rust",
    "Powdery Mildew"
]

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(output)
    confidence = output[predicted_index] * 100

    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    st.success(f"**Disease:** {CLASS_NAMES[predicted_index]}")
    st.info(f"**Confidence:** {confidence:.2f}%")
