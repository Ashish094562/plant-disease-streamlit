# 🌱 Plant Disease Recognition using Deep Learning & TFLite

An end-to-end AI/ML project that detects plant diseases from leaf images using a CNN with EfficientNet (Transfer Learning), optimized using TensorFlow Lite (FLOAT32) and deployed as a real-time web app with Streamlit on Hugging Face Spaces and Streamlit Cloud.

🔗 **Live Demo (Hugging Face):**  
https://huggingface.co/spaces/Ashish094562/Plant_disease_recog

🔗 **Live Demo (Streamlit Cloud):**  
https://plant-disease-app-6ad3rjqvcy9eqq2n9bwemj.streamlit.app/

---

## 🚀 Project Highlights

- CNN-based image classification for 39 plant disease & healthy classes
- Achieved ~99.1% accuracy with strong macro-averaged precision, recall, and F1
- Low-latency inference (~43 ms mean, ~78 ms P95) on CPU using TFLite
- Transfer learning with EfficientNet, fine-tuned for optimal performance
- TensorFlow Lite (FLOAT32) for deployment stability and correctness
- Real-time predictions via Streamlit UI
- Designed for CPU-based & edge-ready deployment
- INT8 quantization planned for further size reduction and faster inference

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Deep Learning:** TensorFlow, Keras, TFLite  
- **Computer Vision:** CNN, Image Classification  
- **Frontend:** Streamlit  
- **Deployment:** Streamlit cloud, Hugging Face Spaces, Hugging Face Hub  
- **Tools:** Git, GitHub  

---

## 🧠 Model & Performance

- **Model Type:** Convolutional Neural Network (CNN)
- **Backbone**: EfficientNet (Transfer Learning)
- **Number of Classes:** 38 (Plant Disease Categories)
-  **Input Size**: 160 × 160 × 3
-  **Training Strategy**:
  - Frozen backbone followed by fine-tuning
  - Best performance achieved during epochs 21–30
- **Validation Accuracy:** 99.1%
- **Optimization**:
  - Converted to TensorFlow Lite (FLOAT32)
  - INT8 quantization planned for edge devices

---

## 📊 Model Performance Metrics
```json
{
  "Accuracy": "~99.1%",
  "Precision (macro)": "~0.989",
  "Recall (macro)": "~0.989",
  "F1-score (macro)": "~0.991",
  "Mean latency": "~43 ms (CPU)",
  "P95 latency": "~78 ms (CPU)",
  "Number of classes": 39,
  "Model format": "TFLite FLOAT32"
}
```
## 📂 Project Structure

```text
plant-disease-streamlit/
├── app.py                           # Streamlit frontend application
├── requirements.txt                 # Python dependencies
├── plant_model_float32.tflite       # TensorFlow Lite FLOAT32 model
├── Plant_Disease_Recognition.ipynb  # Jupyter Notebook
├── label.json                       # labels
├── plant_disease.json               # Disease causes & treatment information
└── README.md                        # Project documentation
└── runtime.txt                      # Set Python version to 3.10 for TensorFlow compatibility
```
---

## ▶️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/Ashish094562/plant-disease-streamlit.git
cd plant-disease-streamlit
```
2. Environment (windows)
```
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

3. Requirements setup
```
pip install -r requirements.txt
```
4. Run the application
   -it will take some time in first try ...
```
streamlit run app.py

```

🌐 Deployment

Model hosted on Hugging Face Hub

Application deployed using Huggingface Spaces and Streamlit cloud

📌 Use Cases

Smart agriculture systems

Automated crop disease detection

Farmer-friendly diagnostic tools

Edge and low-resource ML deployment

👨‍💻 Author

Ashish Singh
Final Year B.Tech | AI / Machine Learning
