# 🐱🐶 Cat vs Dog Classification System  

A complete **end-to-end Modular Deep Learning project** that performs **binary image classification (Cat vs Dog)** using **multiple CNN architectures**.  
The project compares **custom-built and pre-trained models**, selects the **best-performing model**, and deploys it using a **Streamlit web application** on **Streamlit Cloud**.

🏆 **Best Model:** ResNet-18  
📈 **Test Accuracy:** 99%  
🧠 **Models Used:** LeNet-5 (from scratch), AlexNet, ResNet-18  

---

- 🌐 **Live Demo:** [Cat vs Dog Classification System](https://student-performance-indicator-project.onrender.com)

---

## 🎥 Demo

![App Demo](assets/demo.gif)

---

## 📌 Table of Contents
- [Why This Project Exists](#why-this-project-exists)
- [Project Highlights](#project-highlights)
- [Models Used](#models-used)
- [Workflow](#workflow)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Deployment](#deployment)

---

## Why This Project Exists

Most beginner-level Cat vs Dog projects:
- Use **only one pre-trained model**
- Treat CNNs as a **black box**
- Lack **modular code structure**
- Are **not deployment-ready**

This project was built to go beyond that.

### 🎯 Goals of This Project
- Understand **CNN architectures deeply** by implementing **LeNet-5 from scratch**
- Compare **multiple CNN models** fairly under the same training conditions
- Apply **software engineering best practices** (modularity, reusability)
- Deploy a **production-style Deep Learning application**

📌 **Focus:** Learning-oriented, scalable, and resume-ready Deep Learning project.

---

## 🚀 Project Highlights

### 🧠 Deep Learning & Model Comparison
- Implemented **LeNet-5 CNN architecture from scratch**
- Used **AlexNet** and **ResNet-18** for comparison
- All models trained for **2 epochs only** to ensure fair comparison
- Best model selected based on **validation accuracy**

### 🏆 Best Model Selection
- ResNet-18 achieved **99% accuracy**
- Automatically selected as the final inference model

### ⚙️ Modular Codebase
- Separate modules for:
  - Model definitions
  - Training logic
  - Evaluation
  - Inference
- Clean, readable, and scalable structure

### 🌐 Streamlit Web Application
- Interactive UI for image upload
- Real-time prediction (Cat or Dog)
- Deployed on **Streamlit Cloud**
- Lightweight and user-friendly interface

---

## 🧠 Models Used

### 1️⃣ LeNet-5 (From Scratch)
- Fully implemented CNN architecture
- Custom convolutional, pooling, and fully connected layers
- Built for learning and architectural understanding

### 2️⃣ AlexNet
- Deeper CNN with higher representational capacity
- Used for performance comparison

### 3️⃣ ResNet-18 (Best Model)
- Residual connections to solve vanishing gradient problem
- Achieved **99% accuracy**
- Selected as final production model

---

## 🔁 Workflow

1. **Data Loading**
   - Images loaded from train/test directories
   - Standard preprocessing applied

2. **Model Training**
   - Train LeNet-5, AlexNet, and ResNet-18
   - Fixed training epochs (2) for fair comparison

3. **Evaluation**
   - Accuracy calculated for each model
   - Best model selected automatically

4. **Inference**
   - Saved best model weights
   - Used for real-time prediction in Streamlit app

5. **Deployment**
   - Streamlit UI deployed on Streamlit Cloud

---

## 🛠️ Technologies Used

### **Deep Learning & Programming**
- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**

### **Model Architectures**
- LeNet-5 (Custom implementation)
- AlexNet
- ResNet-18

### **Web & Deployment**
- **Streamlit** – UI development
- **Streamlit Cloud** – Deployment platform

### **Tools**
- **Jupyter Notebook** – Experiments & training
- **Git & GitHub** – Version control

---

## 📁 Project Structure


CAT VS DOG CLASSIFIER/ <br>
│ <br>
├── configs/ <br>
│ <br>
├── models/ <br>
│   ├── __init__.py <br>
│   ├── lenet5.py <br>
│   └── model_factory.py    ← Factory of models (LeNet, alexnet, ResNet) <br>
│ <br>
├── NoteBook/ <br>
│   ├── Cat_vs_Dog_Classifier.ipynb <br>
│ <br>
├── training/ <br>
│   ├── __init__.py <br>
│   ├── evaluate.py <br>
│   ├── train_all_models.py <br>
│   └── train_utils.py <br>
│ <br>
│ <br>
├── .gitignore <br>
├── app.py               ← Application entry point <br>
├── inference.py <br>
├── README.md <br>
└── requirements.txt <br>

---

## Setup Instructions

### Prerequisites

To run this project locally, ensure you have the following installed:
- Python 3.8+
- Git
- Virtual environemnt (recommended)

<br>

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Umer2900/Student-Performance-Indicator-Project
   cd Student-Performance-Project
   ```

2. **Install Dependencies**:
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   ```

3. **Install Streamlit App**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   The app will be available at 🌐 `http://localhost:8501`.

<br>
