# EfficienNetB0-ViT-Reasoning


# 🧠 Skin Cancer Detection using EfficientNet-B0 + Vision Transformer

## 📌 Overview

This project presents a hybrid deep learning model combining **EfficientNet-B0 (CNN)** and **Vision Transformer (ViT)** to classify skin lesions from dermatoscopic images.  

The model captures:
- **Local features** (textures, edges)
- **Global patterns** (symmetry, structure)

📄 Accepted at **ICOEI 2026 (International Conference on Electronics and Informatics)**  

---

## 🚀 Key Features

- Hybrid CNN + Transformer architecture  
- Handles class imbalance using **WeightedRandomSampler**  
- Real-time prediction using **Streamlit (SkinScan AI)**  
- Multi-class classification (7 skin disease categories)  
- Provides confidence score and risk assessment  

---

## 📊 Dataset

- **HAM10000 Dataset** (10,000+ images)  
- Classes:
1 . akiec (Actinic keratoses)
2 . bcc (Basal cell carcinoma)
3 . bkl (Benign keratosis)
4 . df (Dermatofibroma)
5 . mel (Melanoma)
6 . nv (Melanocytic nevi)
7 . vasc (Vascular lesions) 

---

## 🧠 Model Architecture

![Hybrid Model Architecture](./assets/model_architecture.png)

### 🔹 Architecture Flow

- **EfficientNet-B0**
  - Extracts local spatial features  

- **Feature Alignment Layer**
  - Converts feature maps into sequence format  

- **Vision Transformer (ViT)**
  - Captures global relationships using attention  

- **Classification Head**
  - Outputs probabilities for 7 classes  

---

## 🛠️ Execution Workflow

### 🔹 Step 1: Data Preparation
- Resize images to **224 × 224**  
- Normalize and convert to tensors  

### 🔹 Step 2: Model Training
```bash
python train.py
````

* Optimizer: **AdamW (lr = 5e-5)**
* Loss: **CrossEntropyLoss**

### 🔹 Step 3: Evaluation

```bash
python evaluate.py
```

* Metrics:

  * Accuracy
  * Recall
  * F1-score
* Generates confusion matrix

### 🔹 Step 4: Run Application

```bash
streamlit run app/app.py
```

---

## 📊 Results

* **Accuracy:** 87.3%
* **F1-Score (Macro):** 78.6%
* **Recall (Macro):** 74.7%

✔ Improved performance on minority classes

---

## 🖥️ Demo

### 🔍 Prediction Output

![Prediction](outputs/sample_output.png)

### 📊 Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/JHANSI_07_10/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Skin-Cancer-Detection/
│
├── app/          # Streamlit app
├── models/       # Model definition
├── utils/        # Dataset & helpers
├── scripts/      # Training & evaluation
├── outputs/      # Results
└── README.md
```

---

## 💡 Future Improvements

* Cloud deployment (AWS / Hugging Face)
* Mobile application integration
* Explainable AI (Grad-CAM)
* Larger and more diverse dataset

---

## 👩‍💻 Authors

* Jhansi Lakshmi Mandugula
* Shaik Arshiya Tarannum
* Golla Mounika
* Harshini Vidapanakallu

```
