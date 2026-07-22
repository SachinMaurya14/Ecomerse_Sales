import os
import joblib
import pandas as pd
import numpy as np

def save_feature_names():
    filename = "feature_names.pkl"
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return

    # Check if best_model.pkl exists and load features from it
    model_path = "best_model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_.tolist()
                joblib.dump(feature_names, filename)
                print(f"Created {filename} successfully from best_model.pkl with {len(feature_names)} features.")
                return
        except Exception as e:
            print(f"Could not read features from model: {e}")

    # Fallback to reconstructing from shopkart_sales_dataset.csv
    csv_path = "shopkart_sales_dataset.csv"
    if not os.path.exists(csv_path):
        # Fallback to hardcoded list matching the pre-trained model exactly
        features = [
            'Customer_Age', 'Qty', 'Unit Price', 'Discount', 'Shipping', 'Delivery',
            'Sales', 'Profit', 'Rating', 'Profit_Category', 'Month', 'Year', 'Is_Weekend',
            'Profit_Margin', 'Revenue_per_Item', 'Gender_Encoded', 'City_Chennai',
            'City_Delhi', 'City_Hyderabad', 'City_Jaipur', 'City_Lucknow', 'City_Mumbai',
            'City_Pune', 'Category_Electronics', 'Category_Fashion', 'Category_Furniture',
            'Category_Grocery', 'Category_Sports', 'Category_Unknown', 'Gender_Male'
        ]
        joblib.dump(features, filename)
        print(f"Created {filename} from fallback feature list.")
        return

    # Load dataset
    df = pd.read_csv(csv_path)

    # 1. Clean outliers / invalid values
    df = df.dropna(subset=["Order_ID"])
    df = df.drop_duplicates().reset_index(drop=True)
    df["Qty"] = df["Qty"].fillna(df["Qty"].median())
    df["Category"] = df["Category"].fillna("Unknown")

    df = df.replace(["?", "N/A", "-999", "null"], np.nan)
    df.loc[df["Customer_Age"] < 0, "Customer_Age"] = np.nan
    df.loc[df["Customer_Age"] > 120, "Customer_Age"] = np.nan
    df["Sales"] = df["Sales"].apply(lambda x: max(x, 0) if pd.notnull(x) else x)

    # Convert date column
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Month"] = df["Order_Date"].dt.month
    df["Year"] = df["Order_Date"].dt.year
    df["Is_Weekend"] = df["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Compute Profit Margin and Revenue per Item
    df["Profit_Margin"] = np.where(df["Sales"] > 0, (df["Profit"] / df["Sales"]) * 100, 0)
    df["Revenue_per_Item"] = np.where(df["Qty"] > 0, df["Sales"] / df["Qty"], 0)

    # Categorical string formatting (which converts NaN to 'Nan')
    categorical_cols = ["Gender", "City", "Category", "Profit_Category"]
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().str.title()

    city_mapping = {"Nyc": "New York", "Ny": "New York", "Usa": "United States", "Us": "United States"}
    df["City"] = df["City"].replace(city_mapping)

    # Gender Encoding (standard label encoding mapping: Female=0, Male=1)
    df["Gender_Encoded"] = np.where(df["Gender"] == "Male", 1, 0)

    # Columns to drop
    cols_to_drop = ["Order_ID", "Order_Date", "Profit_Category_Encoded"]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Convert remaining categorical text into numeric indicators
    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, filename)
    print(f"Created {filename} successfully with {len(feature_names)} features.")

if __name__ == "__main__":
    save_feature_names()
