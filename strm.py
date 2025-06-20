import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load saved models and feature list
classifier = joblib.load("models/RandomForest_model.pkl")
regressor = joblib.load("models/GradientBoosting_model.pkl")
clusterer = joblib.load("models/KMeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features_used.pkl")

st.set_page_config(page_title="Clickstream Dashboard", layout="wide")
st.title("🛍️ Clickstream Conversion & Prediction Dashboard")

# Sidebar navigation
analysis_type = st.sidebar.radio("Choose Analysis", [
    "Data Preview",
    "Classification",
    "Regression",
    "Clustering",
    "Visualizations"
])

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("session_features.csv")

df = load_data()

if analysis_type == "Data Preview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

elif analysis_type == "Classification":
    st.subheader("🔍 Predict Conversion (Classification)")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Predict Conversion"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        prediction = classifier.predict(scaled_input)[0]
        proba = classifier.predict_proba(scaled_input)[0][1]

        label = "Converted" if prediction == 1 else "Not Converted"
        st.success(f"Prediction: {label}")
        st.info(f"Probability of Conversion: {proba:.2f}")

elif analysis_type == "Regression":
    st.subheader("💰 Predict Revenue (Regression)")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Predict Revenue"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        revenue = regressor.predict(scaled_input)[0]
        st.success(f"Predicted Revenue: ₹{revenue:.2f}")

elif analysis_type == "Clustering":
    st.subheader("👥 Customer Segmentation (Clustering)")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Find Cluster"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        cluster = clusterer.predict(scaled_input)[0]
        st.success(f"Customer belongs to Cluster: {cluster}")

elif analysis_type == "Visualizations":
    st.subheader("📊 Data Visualizations")
    st.write("Conversion Distribution")
    st.bar_chart(df['converted'].value_counts())

    st.write("Average Revenue by Category Diversity")
    st.bar_chart(df.groupby('category_diversity')['total_spent'].mean())
