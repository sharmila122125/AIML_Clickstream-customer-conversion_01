import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Models and Metadata
# -------------------------------
classifier = joblib.load("models/RandomForest_model.pkl")
regressor = joblib.load("models/GradientBoosting_model.pkl")
clusterer = joblib.load("models/KMeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features_used.pkl")

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Clickstream Dashboard", layout="wide")
st.title(" Clickstream Conversion & Prediction Dashboard")

# -------------------------------
# Sidebar: Navigation Menu
# -------------------------------
analysis_type = st.sidebar.radio("Choose Analysis", [
    "Data Preview",
    "Classification",
    "Regression",
    "Clustering",
    "Visualizations"
])

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("session_features.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'session_features.csv' not found.")
        return pd.DataFrame()

df = load_data()

# -------------------------------
# 1. Data Preview
# -------------------------------
if analysis_type == "Data Preview":
    st.subheader(" Dataset Preview (First 5000 Rows)")
    if not df.empty:
        st.dataframe(df.head(5000), use_container_width=True)
        st.markdown(f"**Total Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {list(df.columns)}")
    else:
        st.warning("No data available to display.")

# -------------------------------
# 2. Classification
# -------------------------------
elif analysis_type == "Classification":
    st.subheader(" Predict Conversion")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Predict Conversion"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        prediction = classifier.predict(scaled_input)[0]
        proba = classifier.predict_proba(scaled_input)[0][1]

        label = "Converted " if prediction == 1 else "Not Converted ❌"
        st.success(f"Prediction: {label}")
        st.info(f"Probability of Conversion: {proba:.2f}")

# -------------------------------
# 3. Regression
# -------------------------------
elif analysis_type == "Regression":
    st.subheader(" Predict Revenue")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Predict Revenue"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        revenue = regressor.predict(scaled_input)[0]
        st.success(f"Predicted Revenue: ₹{revenue:.2f}")

# -------------------------------
# 4. Clustering
# -------------------------------
elif analysis_type == "Clustering":
    st.subheader(" Customer Segmentation")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("Find Cluster"):
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        cluster = clusterer.predict(scaled_input)[0]
        st.success(f"Customer belongs to Cluster: {cluster}")

# -------------------------------
# 5. Visualizations
# -------------------------------
elif analysis_type == "Visualizations":
    st.subheader(" Visual Insights")

    if 'converted' in df.columns:
        st.write("### Conversion Distribution")
        st.bar_chart(df['converted'].value_counts())

    if 'category_diversity' in df.columns and 'total_spent' in df.columns:
        st.write("### Average Revenue by Category Diversity")
        avg_rev = df.groupby('category_diversity')['total_spent'].mean()
        st.bar_chart(avg_rev)

    else:
        st.warning("Columns required for visualization are missing.")
