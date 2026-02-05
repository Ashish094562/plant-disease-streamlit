import gradio as gr
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
REPO_ID = "Ashish094562/plant-disease-tflite"
MODEL_FILE = "plant_model_quant.tflite"
JSON_FILE = "plant_disease.json"
IMAGE_SIZE = 160

# ---------------- DOWNLOAD FILES ----------------
model_path = hf_hub_download(REPO_ID, MODEL_FILE)
json_path = hf_hub_download(REPO_ID, JSON_FILE)

# ---------------- LOAD MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD DISEASE INFO ----------------
with open(json_path, "r") as f:
    disease_list = json.load(f)

DISEASE_INFO = {d["name"]: d for d in disease_list}

LABELS = list(DISEASE_INFO.keys())

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- PREDICT ----------------
def predict(image):
    x = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    idx = int(np.argmax(output))
    conf = float(np.max(output))
    label = LABELS[idx]

    info = DISEASE_INFO[label]
    return (
        label,
        f"{conf*100:.2f}%",
        info["cause"],
        info["cure"]
    )

# ---------------- GRADIO UI ----------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Text(label="Confidence"),
        gr.Text(label="Cause"),
        gr.Text(label="Cure")
    ],
    title="🌱 Plant Disease Detection",
    description="Upload a plant leaf image to detect disease using a lightweight TFLite model."
)

demo.launch()
