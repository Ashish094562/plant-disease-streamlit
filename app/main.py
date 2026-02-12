
import streamlit as st
from PIL import Image
from model_loader import download_assets, load_model, load_json
from inference import predict

st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease using a **TFLite model**.")

# Download assets
MODEL_PATH, DISEASE_PATH, LABELS_PATH = download_assets()

# Load model
interpreter = load_model(MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load JSON
LABELS = load_json(LABELS_PATH)
disease_list = load_json(DISEASE_PATH)
DISEASE_INFO = {d["name"]: d for d in disease_list}

uploaded_file = st.file_uploader(
    "📷 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        label, confidence, info, probs, top3 = predict(
            image,
            interpreter,
            input_details,
            output_details,
            LABELS,
            DISEASE_INFO
        )

    st.success("✅ Prediction Complete")

    st.subheader("🌿 Prediction")
    st.write(f"**Disease:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    if confidence < 0.50:
        st.warning("⚠️ Low confidence — image may be unclear or out of distribution.")

    st.subheader("🦠 Cause")
    st.write(info["cause"])

    st.subheader("💊 Recommended Treatment")
    st.write(info["cure"])

    st.subheader("📊 Top Predictions")
    for idx in top3:
        st.write(f"{LABELS[int(idx)]} → {probs[int(idx)] * 100:.2f}%")