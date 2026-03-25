import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = Path(r"C:\Clg\TekWorks\Datasets\task1_dataset.csv")


def load_and_clean_raw_data():
    """Load, fill NAs, and cap outliers on raw data."""
    df = pd.read_csv(DATA_PATH)
    
    # Fill missing values with median
    numeric_cols = ["income", "loan_amount", "credit_score", "annual_spend"]
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Cap outliers using IQR
    outlier_cols = ["loan_amount", "credit_score", "income", "num_transactions", "annual_spend"]
    for col in outlier_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    
    # Drop date (non-numeric)
    df.drop(columns=["date"], inplace=True, errors="ignore")
    
    return df


def encode_categoricals(df):
    """One-hot encode categorical columns."""
    categorical_cols = ["city", "employment_type", "loan_type"]
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    
    return df


def engineer_features(df):
    """Create ratio, log, and interaction features."""
    df = df.copy()
    
    # Safe divisions
    income_safe = np.where(df["income"] <= 0, 1, df["income"])
    trans_safe = np.where(df["num_transactions"] <= 0, 1, df["num_transactions"])
    
    df["loan_to_income"] = df["loan_amount"] / income_safe
    df["spend_to_income"] = df["annual_spend"] / income_safe
    df["income_per_trans"] = df["income"] / trans_safe
    df["spend_per_trans"] = df["annual_spend"] / trans_safe
    df["loan_per_trans"] = df["loan_amount"] / trans_safe
    df["credit_norm"] = df["credit_score"] / 900.0
    df["age_income_int"] = df["age"] * df["income"]
    df["loan_credit_int"] = df["loan_amount"] * df["credit_score"]
    df["high_risk"] = (df["credit_score"] < 600).astype(int)
    
    # Log transforms
    log_cols = ["income", "loan_amount", "annual_spend", "num_transactions"]
    for col in log_cols:
        df[f"log_{col}"] = np.log1p(df[col])
    
    return df


def add_poly_features(df, poly_cols):
    """Add polynomial features safely."""
    df = df.copy()
    
    # Ensure no NaN in poly_cols before fitting
    poly_data = df[poly_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_array = poly.fit_transform(poly_data)
    poly_names = [f"poly_{name}" for name in poly.get_feature_names_out(poly_cols)]
    poly_df = pd.DataFrame(poly_array, columns=poly_names, index=df.index)
    
    return pd.concat([df, poly_df], axis=1)


def preprocess_data(df):
    """Full preprocessing pipeline."""
    df = encode_categoricals(df)
    df = engineer_features(df)
    
    # Final NaN/Inf cleanup
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Add polynomial features
    poly_cols = ["age", "income", "loan_amount", "credit_score", "num_transactions", "annual_spend"]
    df = add_poly_features(df, poly_cols)
    
    return df


@st.cache_resource
def train_knn_model():
    """Train KNN model once and cache it."""
    # Load and clean
    df = load_and_clean_raw_data()
    
    # Preprocess
    df = preprocess_data(df)
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Identify numeric vs binary columns
    binary_cols = [col for col in X.columns if X[col].isin([0, 1]).all()]
    numeric_cols = [col for col in X.columns if col not in binary_cols]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Train KNN (tuned parameters from notebook)
    knn = KNeighborsRegressor(n_neighbors=31, weights="distance", p=1)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test_scaled)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    
    # Refit on full data for inference
    X_scaled_full = X.copy()
    X_scaled_full[numeric_cols] = scaler.transform(X[numeric_cols])
    knn.fit(X_scaled_full, y)
    
    return {
        "model": knn,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "all_columns": X.columns.tolist(),
        "metrics": metrics,
    }


def make_prediction(user_data, model_dict):
    """Generate prediction for user input."""
    # Create DataFrame from user input
    user_df = pd.DataFrame([user_data])
    
    # Preprocess exactly like training data
    user_df = preprocess_data(user_df)
    
    # Reindex to match training columns
    user_df = user_df.reindex(columns=model_dict["all_columns"], fill_value=0)
    
    # Scale numeric columns
    X_scaled = user_df.copy()
    X_scaled[model_dict["numeric_cols"]] = model_dict["scaler"].transform(
        user_df[model_dict["numeric_cols"]]
    )
    
    # Predict
    prediction = model_dict["model"].predict(X_scaled)[0]
    return prediction


def main():
    st.set_page_config(page_title="KNN Spend Predictor", layout="wide")
    st.title("Customer Spend Prediction (KNN Model)")
    st.caption("Powered by engineered features and tuned KNeighborsRegressor")
    
    # Train model once
    model_dict = train_knn_model()
    
    # Display metrics
    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{model_dict['metrics']['r2']:.4f}")
    c2.metric("MSE", f"{model_dict['metrics']['mse']:,.0f}")
    c3.metric("MAE", f"{model_dict['metrics']['mae']:,.0f}")
    
    # Input form
    st.subheader("Enter Applicant Details")
    with st.form("input_form"):
        age = st.number_input("Age", min_value=18, max_value=80, value=40)
        income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=1_000_000.0, value=120000.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=5000.0, max_value=2_000_000.0, value=250000.0, step=5000.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=680.0)
        num_transactions = st.number_input("Number of Transactions", min_value=1, max_value=500, value=75)
        annual_spend = st.number_input("Annual Spend ($)", min_value=1000.0, max_value=3_000_000.0, value=220000.0, step=5000.0)
        
        # Categorical inputs
        city_options = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"]
        city = st.selectbox("City", options=city_options)
        
        emp_options = ["Salaried", "Self-Employed", "Student", "Unemployed"]
        employment_type = st.selectbox("Employment Type", options=emp_options)
        
        loan_options = ["Auto", "Education", "Home", "Personal"]
        loan_type = st.selectbox("Loan Type", options=loan_options)
        
        submitted = st.form_submit_button("Predict Target")
    
    if submitted:
        user_input = {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "num_transactions": num_transactions,
            "annual_spend": annual_spend,
            "city": city,
            "employment_type": employment_type,
            "loan_type": loan_type,
        }
        
        try:
            prediction = make_prediction(user_input, model_dict)
            st.success(f"**Predicted Target Value: ${prediction:,.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    st.divider()
    with st.expander("📊 View Sample Data"):
        raw_df = load_and_clean_raw_data()
        st.dataframe(raw_df.head(10))


if __name__ == "__main__":
    main()
