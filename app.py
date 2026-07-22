import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Profit Category Predictor", page_icon="📈")

@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

st.title("📈 Profit Category Prediction")

try:
    model, scaler = load_assets()
    st.success("Model assets loaded successfully!")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.warning("Make sure 'best_model.pkl' and 'scaler.pkl' are in this folder.")
