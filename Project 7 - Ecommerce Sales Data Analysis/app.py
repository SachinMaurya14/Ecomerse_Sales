import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration for a premium, professional UI
st.set_page_config(
    page_title="ShopKart Profit Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .metric-card-low {
        border-left-color: #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# Load resources with st.cache_resource
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model/scaler/features: {e}")
    assets_loaded = False

st.title("📊 ShopKart Order Profitability Predictor")
st.markdown("Predict whether a new order will generate **High Profit** or **Low Profit** before dispatch.")

if assets_loaded:
    # Layout using columns
    col1, col2 = st.columns([2, 1.5], gap="large")

    with col1:
        st.subheader("🛒 Order Information")
        
        # User input fields in grid layout
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            sales = st.number_input("Sales ($)", min_value=0.0, value=1200.0, step=50.0, help="Total sales value of the order")
            qty = st.number_input("Quantity", min_value=1, value=2, step=1, help="Number of items ordered")
            unit_price = st.number_input("Unit Price ($)", min_value=0.0, value=600.0, step=10.0, help="Price per item")
            profit = st.number_input("Profit ($)", value=300.0, step=10.0, help="Expected profit value")
            customer_age = st.slider("Customer Age", min_value=0, max_value=120, value=34, help="Age of the customer")
            
        with grid_col2:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            city = st.selectbox("City", ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Lucknow", "Mumbai", "Pune", "New York", "United States"], index=8)
            category = st.selectbox("Category", ["Electronics", "Beauty", "Grocery", "Furniture", "Fashion", "Sports"], index=0)
            order_date = st.date_input("Order Date", datetime(2026, 3, 15))
            
        # Collapsible section for advanced inputs (features present in the model but not explicitly required in basic input)
        with st.expander("⚙️ Advanced / Additional Fields"):
            discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=20.0, help="Discount applied to the order")
            shipping = st.number_input("Shipping Cost ($)", min_value=0.0, value=316.0, help="Shipping cost for the order")
            delivery = st.number_input("Delivery Time (Days)", min_value=0, value=5, help="Estimated days for delivery")
            rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.5, help="Product/service rating")

    with col2:
        st.subheader("🔮 Prediction & Analysis")
        
        if st.button("Predict Profitability", use_container_width=True):
            # 1. Create a DataFrame from inputs
            input_data = pd.DataFrame([{
                "Customer_Age": customer_age,
                "Gender": gender,
                "City": city,
                "Category": category,
                "Qty": qty,
                "Sales": sales,
                "Profit": profit,
                "Unit Price": unit_price,
                "Discount": discount,
                "Shipping": shipping,
                "Delivery": delivery,
                "Rating": rating,
                "Order_Date": pd.to_datetime(order_date)
            }])
            
            # Apply exact preprocessing steps
            # Date engineering
            input_data["Month"] = input_data["Order_Date"].dt.month
            input_data["Year"] = input_data["Order_Date"].dt.year
            input_data["Is_Weekend"] = input_data["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Profit margin and Revenue per item
            input_data["Profit_Margin"] = np.where(input_data["Sales"] > 0, (input_data["Profit"] / input_data["Sales"]) * 100, 0)
            input_data["Revenue_per_Item"] = np.where(input_data["Qty"] > 0, input_data["Sales"] / input_data["Qty"], 0)
            
            # Standardize text variables
            input_data["City"] = input_data["City"].astype(str).str.strip().str.title()
            input_data["Category"] = input_data["Category"].astype(str).str.strip().str.title()
            input_data["Gender"] = input_data["Gender"].astype(str).str.strip().str.title()
            
            # Gender Encoded
            input_data["Gender_Encoded"] = np.where(input_data["Gender"] == "Male", 1, 0)
            
            # Map Profit Category threshold (Profit > 3000 -> 1 else 0)
            input_data["Profit_Category"] = np.where(input_data["Profit"] > 3000, 1, 0)
            
            # Drop unnecessary columns
            cols_to_drop = ["Order_ID", "Order_Date"]
            processed_data = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns])
            
            # One-hot encode categoricals
            encoded_data = pd.get_dummies(processed_data, drop_first=True)
            
            # Align features with feature_names.pkl
            aligned_data = encoded_data.reindex(columns=feature_names, fill_value=0)
            
            # Scale numeric features using loaded scaler
            scale_cols = ["Sales", "Profit", "Qty", "Unit Price"]
            aligned_data[scale_cols] = scaler.transform(aligned_data[scale_cols])
            
            # Make prediction
            prediction = model.predict(aligned_data)[0]
            probabilities = model.predict_proba(aligned_data)[0] if hasattr(model, "predict_proba") else None
            
            # Map predictions to class labels
            class_labels = {0: "Low Profit", 1: "High Profit"}
            result_label = class_labels.get(prediction, "Unknown")
            
            # Render prediction card
            card_class = "metric-card" if prediction == 1 else "metric-card metric-card-low"
            color_text = "#4CAF50" if prediction == 1 else "#f44336"
            
            st.markdown(f"""
                <div class="{card_class}">
                    <h4 style='margin: 0; color: #555;'>Prediction Result</h4>
                    <h2 style='margin: 0.5rem 0; color: {color_text};'>{result_label}</h2>
                    <p style='margin: 0; color: #777;'>Class Code: {prediction}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if probabilities is not None:
                st.markdown("### 📈 Confidence Probabilities")
                for idx, prob in enumerate(probabilities):
                    lbl = class_labels.get(idx, f"Class {idx}")
                    st.write(f"**{lbl}** ({prob*100:.1f}%)")
                    st.progress(float(prob))
            
            # Display Preprocessed feature values for debugging/clarity
            st.markdown("### ⚙️ Computed Engineering Features")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Profit Margin (%)", f"{input_data['Profit_Margin'].values[0]:.2f}%")
                st.metric("Revenue per Item ($)", f"${input_data['Revenue_per_Item'].values[0]:.2f}")
            with metrics_col2:
                st.metric("Is Weekend", "Yes" if input_data['Is_Weekend'].values[0] == 1 else "No")
                st.metric("Month / Year", f"{input_data['Month'].values[0]} / {input_data['Year'].values[0]}")
        else:
            st.info("Click the **Predict Profitability** button to run inference.")
else:
    st.warning("Please run `save_assets.py` first to generate feature artifacts.")
