# 📊 ShopKart Order Profitability Predictor

An end-to-end Machine Learning project to predict whether an e-commerce order will generate **High Profit** or **Low Profit** before dispatch. Built using Streamlit, scikit-learn, pandas, and joblib.

This tool helps online retailers like ShopKart identify low-profit orders early, optimize pricing strategies, control discounts, and reduce shipping losses.

---

## 📂 Project Repository Structure

This repository follows industry-standard structure for Machine Learning projects:

```text
.
├── data/
│   └── shopkart_sales_dataset.csv     # Raw e-commerce orders dataset
├── notebooks/
│   └── shopkart_sales_dataset.ipynb   # Jupyter Notebook containing EDA, preprocessing, and model training
├── docs/
│   └── Project Requirements.txt        # Business guidelines and step-by-step specifications
├── app.py                             # Main Streamlit web application
├── save_assets.py                     # Script to export feature list alignment
├── best_model.pkl                     # Serialized best-performing ML model (e.g. Random Forest / Gradient Boosting)
├── scaler.pkl                         # Serialized StandardScaler model
├── feature_names.pkl                  # List of feature names matching model inputs
├── requirements.txt                   # Project dependency versions
├── .gitignore                         # Standard git ignore definitions
└── LICENSE                            # MIT License
```

---

## 🛠️ Tech Stack & Libraries

* **Frontend Dashboard**: Streamlit
* **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, KNN, SVM, Decision Tree)
* **Data Processing & Analytics**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Model Serialization**: Joblib

---

## 🚀 Getting Started

Follow these step-by-step instructions to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\Activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Preprocessing Assets
Run the asset saver to export feature definitions:
```bash
python save_assets.py
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📈 Machine Learning Workflow

The underlying machine learning notebook follows these comprehensive steps:
1. **Exploratory Data Analysis (EDA)**: Profiling variables, checking class balances, correlation matrices, and distribution plots.
2. **Data Cleaning**: Handling missing values, deleting duplicates, and removing outliers using the Interquartile Range (IQR) method.
3. **Feature Engineering**: Creating derived features like `Profit Margin`, `Revenue per Item`, `Month`, `Year`, and `Is_Weekend`.
4. **Data Preparation**: One-hot encoding categorical variables, scaling numeric columns with `StandardScaler`.
5. **Model Evaluation**: Training multiple classifiers and tuning parameters with `GridSearchCV`.
6. **Deployment**: Exporting the best-performing model pipeline for real-time inference on the Streamlit dashboard.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
