
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_final.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

CURRENCIES = {
    "INR — Indian Rupee": {"code": "INR", "rate": 1.0},
    "MAD — Moroccan Dirham": {"code": "MAD", "rate": 0.1007},
    "USD — US Dollar": {"code": "USD", "rate": 0.0103},
    "EUR — Euro": {"code": "EUR", "rate": 0.0092},
    "GBP — British Pound": {"code": "GBP", "rate": 0.0078},
}

# The notebook's saved model was trained in 2026.
TRAINING_REFERENCE_YEAR = 2026

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="wide",
)

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Scaler not found. Save the fitted training scaler to "
            "models/scaler.pkl before running predictions."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


st.title("🚘 Car Price Predictor")
st.write(
    "Estimate the selling price of a used car using an XGBoost "
    "regression model."
)

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error(
        "The model does not contain the expected feature names. "
        "Please verify the saved model."
    )
    st.stop()

features = list(model.feature_names_in_)

# --------------------------------------------------
# Derive options from the trained model
# --------------------------------------------------

brand_options = sorted(
    feature[len("brand_"):]
    for feature in features
    if feature.startswith("brand_")
)

fuel_options = sorted(
    feature[len("fuel_"):]
    for feature in features
    if feature.startswith("fuel_")
)

seller_options = sorted(
    feature[len("seller_type_"):]
    for feature in features
    if feature.startswith("seller_type_")
)

if not brand_options or not fuel_options or not seller_options:
    st.error("Could not extract categorical options from the model.")
    st.stop()

owner_mapping = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

transmission_mapping = {
    "Manual": 0,
    "Automatic": 1,
}

# --------------------------------------------------
# Prepare input using training transformations
# --------------------------------------------------

def prepare_input(
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    brand,
):
    car_age = TRAINING_REFERENCE_YEAR - year

    # Prevent division by zero for a vehicle from the reference year.
    km_per_age = km_driven / car_age if car_age > 0 else 0

    age_km = car_age * km_driven

    # Initialize all expected model features.
    row = {feature: 0 for feature in features}

    # Numeric and engineered features.
    row["year"] = year
    row["km_driven"] = km_driven
    row["car_age"] = car_age
    row["km_per_age"] = km_per_age
    row["age_km"] = age_km

    # Match the notebook's categorical encodings.
    row["transmission"] = transmission_mapping[transmission]

    encoded_owner = owner_mapping[owner]
    row["owner"] = encoded_owner

    # Match: df["age_owner"] = df["car_age"] * df["owner"].replace(0, 1)
    row["age_owner"] = car_age * (
        encoded_owner if encoded_owner != 0 else 1
    )

    # One-hot encoded columns.
    categorical_features = {
        f"fuel_{fuel}": 1,
        f"seller_type_{seller_type}": 1,
        f"brand_{brand}": 1,
    }

    for feature, value in categorical_features.items():
        if feature in row:
            row[feature] = value

    # Preserve the exact order expected by XGBoost.
    input_df = pd.DataFrame([row], columns=features)

    # Apply the same fitted scaler used during training.
    numeric_cols = ["year", "km_driven"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    return input_df


# --------------------------------------------------
# User interface
# --------------------------------------------------

with st.sidebar:
    st.header("Vehicle details")

    brand = st.selectbox("Brand / model", brand_options)

    year = st.number_input(
        "Manufacturing year",
        min_value=1980,
        max_value=datetime.now().year,
        value=min(2018, datetime.now().year),
        step=1,
    )

    km_driven = st.number_input(
        "Kilometers driven",
        min_value=0,
        max_value=1_000_000,
        value=60_000,
        step=1_000,
    )

    fuel = st.selectbox("Fuel type", fuel_options)
    seller_type = st.selectbox("Seller type", seller_options)

    transmission = st.selectbox(
        "Transmission",
        list(transmission_mapping.keys()),
    )

    owner = st.selectbox(
        "Owner history",
        list(owner_mapping.keys()),
    )

    currency_label = st.selectbox(
        "Display currency",
        list(CURRENCIES.keys()),
    )

    selected_currency = CURRENCIES[currency_label]

    predict_clicked = st.button(
        "Predict selling price",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Vehicle summary
# --------------------------------------------------

st.subheader("Vehicle summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Brand / model", brand)
col2.metric("Year", str(year))
col3.metric("Mileage", f"{km_driven:,} km")
col4.metric(
    "Vehicle age",
    f"{TRAINING_REFERENCE_YEAR - year} years",
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_clicked:
    try:
        input_df = prepare_input(
            year=int(year),
            km_driven=int(km_driven),
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            brand=brand,
        )
        
        prediction = float(model.predict(input_df)[0])

        converted_price = prediction * selected_currency["rate"]
        currency_code = selected_currency["code"]

        st.divider()
        st.subheader("Estimated selling price")

        st.metric(
            label=f"Predicted price ({currency_code})",
            value=f"{converted_price:,.2f} {currency_code}",
        )

        if currency_code != "INR":
            st.caption(
                f"Original model prediction: {prediction:,.2f} INR. "
                "Converted using an illustrative exchange rate, not a live quote."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# Model information
# --------------------------------------------------

with st.expander("Model information"):
    st.write(f"**Model:** {type(model).__name__}")
    st.write(f"**Expected features:** {len(features)}")
    st.write("**Scaler:** StandardScaler")
    st.write(f"**Feature reference year:** {TRAINING_REFERENCE_YEAR}")
