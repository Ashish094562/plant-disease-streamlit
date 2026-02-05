import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌱 Plant Disease Recognition")
st.write("Upload a leaf image to predict the disease")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="plant_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"🌿 Predicted Class: {class_id}")
    st.info(f"Confidence: {confidence:.2f}%")
