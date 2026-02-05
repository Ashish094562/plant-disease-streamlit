# 🌱 Plant Disease Recognition using Deep Learning & TFLite

An end-to-end **AI/ML project** that detects plant diseases from leaf images using a **Convolutional Neural Network (CNN)**, optimized with **TFLite INT8 quantization**, and deployed using **Streamlit** on **Hugging Face Spaces** and **Streamlit cloud**.

🔗 **Live Demo (Hugging Face):**  
https://huggingface.co/spaces/Ashish094562/Plant_disease_recog  

🔗 **Live Demo (Streamlit Cloud):**  
https://plant-disease-app-rnkhccrzvbnrlfv3lkwequ.streamlit.app/

---

## 🚀 Project Highlights

- CNN-based image classification for **38 plant disease categories**
- Achieved **~95% validation accuracy** with strong class-wise performance
- **TFLite INT8 quantization** reduced model size by **~75%**
- **Low-latency inference (<100 ms)** per image on CPU
- Real-time predictions using **Streamlit**
- Lightweight and suitable for **edge & low-resource devices**

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
- **Number of Classes:** 38  
- **Validation Accuracy:** ~95%  
- **F1-Score:** ~0.94  
- **Inference Time:** <100 ms (CPU)  
- **Model Size Reduction:** ~75% after INT8 quantization
- **Techniques:** Transfer Learning

---

## 📂 Project Structure

```text
plant-disease-streamlit/
├── app.py                    # Streamlit frontend application
├── requirements.txt          # Python dependencies
├── plant_model_quant.tflite  # Quantized TFLite model
├── plant_disease.json        # Disease causes & treatment information
└── README.md                 # Project documentation
```
---

## ▶️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/Ashish094562/plant-disease-streamlit.git
cd plant-disease-streamlit
```
2. Environment
```
python -m venv myvenv
```
3. Environment Activate
   -for windows(PowerShell) :- .\myvenv\Scripts\Activate.ps1   or  for mac/linux :- source myvenv/bin/activate

```
\myvenv\Scripts\Activate.ps1
```
4. Requirements setup
```
pip install -r requirements.txt
```
5. Run
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
