import streamlit as st
import pandas as pd
import joblib
import os

# Load models
classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
clusterer = joblib.load("models/clusterer.pkl")

st.set_page_config(page_title="Clickstream App", layout="wide")
st.title("Clickstream Conversion & Prediction Dashboard")

# Load session features from CSV
@st.cache_data
def load_data():
    return pd.read_csv("session_features.csv")

df = load_data()

# Sidebar navigation
section = st.sidebar.radio("Go to", [
    "Session Features Preview",
    "Conversion Prediction (Classification)",
    "Revenue Estimation (Regression)",
    "Customer Segmentation (Clustering)",
    "Segment Distribution",
    "Full Prediction Table"
])

# Section 1: Data Preview
if section == "Session Features Preview":
    st.subheader("📊 Session Features Preview")
    st.dataframe(df.head(50))

# Section 2: Classification
elif section == "Conversion Prediction (Classification)":
    st.subheader("🔍 Conversion Prediction")
    input_data = df.drop(columns=["session_id", "converted"], errors="ignore")
    predictions = classifier.predict(input_data)
    df["Predicted_Conversion"] = predictions
    st.dataframe(df[["session_id", "Predicted_Conversion"]].head(50))

# Section 3: Regression
elif section == "Revenue Estimation (Regression)":
    st.subheader("💰 Revenue Estimation")
    input_data = df.drop(columns=["session_id", "converted"], errors="ignore")
    predicted_revenue = regressor.predict(input_data)
    df["Estimated_Revenue"] = predicted_revenue
    st.dataframe(df[["session_id", "Estimated_Revenue"]].head(50))

# Section 4: Clustering
elif section == "Customer Segmentation (Clustering)":
    st.subheader("👥 Customer Segmentation")
    input_data = df.drop(columns=["session_id", "converted"], errors="ignore")
    clusters = clusterer.predict(input_data)
    df["Predicted_Segment"] = clusters
    st.dataframe(df[["session_id", "Predicted_Segment"]].head(50))

# Section 5: Segment Distribution
elif section == "Segment Distribution":
    st.subheader("📈 Segment Distribution")
    if "Predicted_Segment" not in df.columns:
        input_data = df.drop(columns=["session_id", "converted"], errors="ignore")
        df["Predicted_Segment"] = clusterer.predict(input_data)
    segment_count = df["Predicted_Segment"].value_counts().reset_index()
    segment_count.columns = ["Segment", "Count"]
    st.bar_chart(segment_count.set_index("Segment"))

# Section 6: Full Prediction Table
elif section == "Full Prediction Table":
    st.subheader("Full Prediction Results")
    input_data = df.drop(columns=["session_id", "converted"], errors="ignore")

    df["Predicted_Conversion"] = classifier.predict(input_data)
    df["Estimated_Revenue"] = regressor.predict(input_data)
    df["Predicted_Segment"] = clusterer.predict(input_data)

    st.dataframe(df.head(100))

    # Optionally add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Prediction CSV", csv, "full_predictions.csv", "text/csv")
