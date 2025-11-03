# 🐮 AI-Powered Indian Cattle Breed Identifier

This project provides a **robust AI-powered image classification system** for identifying various **Indian cattle and buffalo breeds**, complete with a user-friendly **Streamlit web application** for real-time prediction and detailed breed information.

The classification model is based on a **fine-tuned ResNet-50 architecture**, leveraging **transfer learning on ImageNet weights** for high accuracy and reliability.

---

## 🌟 Features

- 🖼️ **Real-time Prediction** – Upload an image or use your camera to instantly classify the cattle breed.  
- 📊 **Top-3 Confidence Scores** – Displays the top 3 predicted breeds with confidence percentages.  
- 📘 **Detailed Breed Information** – Includes origin, primary use, key traits, and milk yield for 30+ Indian breeds (loaded from `config.py`).  
- 🔊 **Voice Output** – Hear the predicted breed name and a brief description using the “Read Aloud” feature.  
- 🧠 **Transfer Learning** – Fine-tuned **ResNet-50** model for high accuracy.  
- ⚡ **User-Friendly Web App** – Built using **Streamlit** for an intuitive interface.  

---

## 💻 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | The main **Streamlit** web application for the user interface and predictions. |
| `train.py` | Script for training the **ResNet-50** model, including data splitting and checkpoint saving. |
| `predict.py` | Script containing the core **predict function** for inference on a single image. |
| `model.py` | Defines the **ResNet-50** model architecture and custom classifier head. |
| `dataset.py` | Custom **PyTorch Dataset** class for handling and transforming images. |
| `config.py` | Stores metadata for all supported breeds (`BREED_INFO` dictionary). |
| `best_model.pth` | Saved PyTorch model checkpoint (best-performing model). |
| `Cattle` | Folder containing dataset images, organized by breed folders. |

---

## ⚙️ Installation

### 🧩 Prerequisites
- Python **3.8+**
- Basic knowledge of PyTorch and Streamlit (optional)

### 🛠️ Setup

1. **Clone the repository:**
   ```bash
   Copy the Cattle Breed Classifier folder into your system.

2. **Install Dependencies:**
   pip install torch torchvision torchaudio
   pip install streamlit scikit-learn matplotlib Pillow gTTS tqdm


## 🚀 Usage

1. **Training the Model:**
   python train.py
   Note: Run this if you want to train the model again if not directly run the app.py file.

2. **Making Predictions:**
   python predict.py

3. **Running the Web App:**
   python -m streamlit run app.py

Team Name: Epoch Makers

Team coordinator name: Subal

Team coordinator phone number:

Project Theme: AI in Agriculture 

Project Title: AI-Powered Breed Identification for Indian Livestock
