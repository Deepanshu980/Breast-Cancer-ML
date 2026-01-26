# Breast Cancer Prediction using Machine Learning 🎗️🧠

This project uses **Machine Learning** to predict whether a breast tumor is **benign or malignant** based on diagnostic features.  
The goal is to assist early detection and improve decision-making in healthcare.

---

## 📌 Problem Statement

Breast cancer is one of the most common cancers worldwide.  
Early and accurate diagnosis significantly increases survival rates.

Manual diagnosis:
- Is time-consuming
- Depends heavily on expert availability
- Can be prone to human error

👉 This project builds a **data-driven ML model** to automate and improve prediction accuracy.

---

## 🧠 Solution Overview

We train and evaluate multiple **machine learning classifiers** on breast cancer diagnostic data to:
- Classify tumors as **Benign (B)** or **Malignant (M)**
- Achieve **high accuracy and reliability**
- Reduce false negatives (critical in healthcare)

---

## 📂 Dataset Description

The dataset contains **cell nucleus features** computed from breast mass images.

### Key Features (examples)

| Feature | Description |
|-------|------------|
| `radius_mean` | Mean radius of tumor |
| `texture_mean` | Mean texture |
| `perimeter_mean` | Mean perimeter |
| `area_mean` | Mean area |
| `smoothness_mean` | Mean smoothness |
| `compactness_mean` | Mean compactness |
| `concavity_mean` | Mean concavity |
| `concave_points_mean` | Mean concave points |
| `symmetry_mean` | Mean symmetry |
| `fractal_dimension_mean` | Mean fractal dimension |

### Target Variable
- `0` → Benign  
- `1` → Malignant  

---

## ⚙️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Jupyter Notebook**

---

## 🔍 Machine Learning Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree

👉 Best performing model is selected based on evaluation metrics.

---

## 📈 Model Performance

| Metric | Score |
|------|------|
| Accuracy | **97%+** |
| Precision | High |
| Recall | High |
| F1-Score | Balanced |

*(Exact results may vary depending on model and tuning)*

---

## 🛠️ Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Final Prediction

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-ml.git
cd breast-cancer-ml
