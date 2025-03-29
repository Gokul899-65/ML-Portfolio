import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('logistic_model.pkl', 'rb') as f:
    data = pickle.load(f)
    theta = data['theta']
    scaler = data['scaler']

# Logistic function
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# Predict function
def predict(features_scaled):
    features_scaled = np.append([1], features_scaled)  # add intercept
    probability = logistic_function(np.dot(features_scaled, theta))
    prediction = int(probability >= 0.5)
    return prediction, probability[0]

# Streamlit app
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("Breast Cancer Prediction App")
st.write("This app uses a custom logistic regression model to predict whether a tumor is **benign** or **malignant**.")

# Feature input
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.sidebar.header("Input Features")
input_data = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction, probability = predict(input_scaled)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The tumor is **Benign** with probability {probability:.2f}")
    else:
        st.error(f"The tumor is **Malignant** with probability {probability:.2f}")
