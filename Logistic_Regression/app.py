import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('Logistic_Regression/logistic_model.pkl', 'rb') as f:
    data = pickle.load(f)
    theta = data['theta']
    scaler = data['scaler']

# Logistic function
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

def predict(features_scaled):
    features_scaled = np.append([1], features_scaled)  # add intercept
    probability = logistic_function(np.dot(features_scaled, theta))
    prediction = int(probability >= 0.5)
    return prediction, probability[0]

# Top 5 most important features
top_features = [
    'worst perimeter',
    'worst concave points',
    'mean concave points',
    'worst radius',
    'mean perimeter'
]

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🩺 Simplified Breast Cancer Prediction")
st.write("Only the most important features are needed to predict tumor type.")

st.sidebar.header("🔧 Input Features")
input_data = []
for feature in top_features:
    val = st.sidebar.number_input(f"{feature}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)

    # Fill out 30 features with zeros, and insert only top 5 at correct indices
    full_input = np.zeros((1, 30))  # total 30 features
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    for i, feature in enumerate(top_features):
        idx = list(feature_names).index(feature)
        full_input[0, idx] = input_array[0, i]

    input_scaled = scaler.transform(full_input)
    prediction, probability = predict(input_scaled)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ The tumor is **Begining** with probability {probability:.2f}")
    else:
        st.error(f"⚠️ The tumor is **Malignant** with probability {probability:.2f}")
