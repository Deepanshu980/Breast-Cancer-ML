import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_joblib.pkl")

# Page configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

# Sidebar
st.sidebar.title("📌 Instructions")
st.sidebar.markdown(
    """
    1. Enter all the **medical test values** in the input fields.  
    2. Click **Run Prediction** to analyze the data.  
    3. The model will predict whether the tumor is **Malignant** or **Benign**.  

    ---
    ⚠️ **Note**: This tool is for **educational purposes only** and should not be used as a substitute for medical advice.
    """
)

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app uses a **Machine Learning model** trained on the 
    [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).
    
    - ✅ Benign → Non-Cancerous  
    - 🚨 Malignant → Cancerous  
    """
)

# Main title
st.title("🎗️ Breast Cancer Prediction App")

st.markdown(
    """
    This tool helps predict whether a tumor is **Malignant (Cancerous)** or 
    **Benign (Non-Cancerous)** based on **30 medical features**.  
    Fill in the details below and click **Run Prediction**.
    """
)

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Input section
st.subheader("📝 Enter Patient Measurements")

cols = st.columns(3)
inputs = []

for idx, feature in enumerate(feature_names):
    with cols[idx % 3]:
        val = st.number_input(f"{feature}", value=1.0, format="%.3f")
        inputs.append(val)

features = np.array(inputs).reshape(1, -1)

# Prediction button
st.markdown("---")
if st.button("🔍 Run Prediction", use_container_width=True):
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.markdown(
            """
            <div style="padding:20px; border-radius:10px; background-color:#ffe5e5; border:2px solid #ff4d4d">
            <h2 style="color:#cc0000">🚨 Prediction: Malignant (Cancerous)</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding:20px; border-radius:10px; background-color:#e6ffe6; border:2px solid #00cc66">
            <h2 style="color:#006600">✅ Prediction: Benign (Non-Cancerous)</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input summary
    with st.expander("📊 View Input Summary"):
        st.write({feature: val for feature, val in zip(feature_names, inputs)})

# Footer
st.markdown(
    """
    <hr style="border:1px solid #ddd">
    <div style="text-align:center; color:gray; font-size:14px">
    Built with ❤️ using <b>Streamlit</b> | Breast Cancer Prediction App
    </div>
    """,
    unsafe_allow_html=True,
)
