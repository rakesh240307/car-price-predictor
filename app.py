
import streamlit as st
import pandas as pd
import joblib
import os

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "xgboost_final_v1.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR, "models", "preprocessor_v1.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "car_price_model_data_v1.csv"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


model, preprocessor = load_model()
df = load_data()

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
        border: 1px solid #374151;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.15rem;
        color: #d1d5db;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #064e3b,
            #065f46
        );
        border: 1px solid #10b981;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        color: #a7f3d0;
        font-size: 1rem;
    }

    .result-price {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-sub {
        color: #d1fae5;
    }

    .info-card {
        padding: 1.2rem;
        border-radius: 15px;
        background: #111827;
        border: 1px solid #374151;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# HERO
# ==================================================

st.markdown(
    """
<div class="hero">
<h1>🚗 Car Price Predictor</h1>
<p>Estimate the selling price of a used car using a machine-learning model trained on real-world used-car data.</p>
</div>
""",
    unsafe_allow_html=True
)

# ==================================================
# DATA OPTIONS
# ==================================================

brands = sorted(df["brand"].unique())

fuel_types = sorted(
    df["fuel_type"].unique()
)

transmissions = sorted(
    df["transmission_type"].unique()
)

seller_types = sorted(
    df["seller_type"].unique()
)

# ==================================================
# CAR DETAILS
# ==================================================

st.markdown(
    '<div class="section-title">🚘 Car Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
        "Brand",
        brands
    )

with col2:

    models = sorted(
        df[df["brand"] == brand]["model"].unique()
    )

    model_name = st.selectbox(
        "Model",
        models
    )

# ==================================================
# SPECIFICATIONS
# ==================================================

st.markdown(
    '<div class="section-title">⚙️ Vehicle Specifications</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    vehicle_age = st.number_input(
        "Vehicle Age (years)",
        min_value=0,
        max_value=30,
        value=3,
        step=1
    )

with c2:

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=1_000_000,
        value=40_000,
        step=1_000
    )

with c3:

    mileage = st.number_input(
        "Mileage (km/l)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.1
    )

c4, c5, c6 = st.columns(3)

with c4:

    engine = st.number_input(
        "Engine (cc)",
        min_value=500,
        max_value=10_000,
        value=1500,
        step=50
    )

with c5:

    max_power = st.number_input(
        "Max Power (bhp)",
        min_value=0.0,
        max_value=2_000.0,
        value=110.0,
        step=1.0
    )

with c6:

    seats = st.number_input(
        "Seats",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )

# ==================================================
# OTHER DETAILS
# ==================================================

st.markdown(
    '<div class="section-title">🔧 Other Details</div>',
    unsafe_allow_html=True
)

c7, c8, c9 = st.columns(3)

with c7:

    seller_type = st.selectbox(
        "Seller Type",
        seller_types
    )

with c8:

    fuel_type = st.selectbox(
        "Fuel Type",
        fuel_types
    )

with c9:

    transmission_type = st.selectbox(
        "Transmission",
        transmissions
    )

st.divider()

# ==================================================
# BUTTONS
# ==================================================

predict_col, reset_col = st.columns([3, 1])

with predict_col:

    predict_button = st.button(
        "💰 Predict Car Price",
        use_container_width=True
    )

with reset_col:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )

# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    input_data = pd.DataFrame([{

        "brand": brand,
        "model": model_name,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats

    }])

    # Preprocess
    input_processed = preprocessor.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        input_processed
    )[0]

    prediction = max(0, prediction)

    # Convert price
    if prediction >= 10_000_000:

        price_display = (
            f"₹{prediction / 10_000_000:.2f} Crore"
        )

    else:

        price_display = (
            f"₹{prediction / 100_000:.2f} Lakh"
        )

    # Result
    st.markdown(
        f"""
<div class="result-card">
<div class="result-label">Estimated Selling Price</div>
<div class="result-price">{price_display}</div>
<div class="result-sub">Approx. ₹{prediction:,.0f}</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.success(
        "Prediction generated successfully!"
    )

# ==================================================
# MODEL INFORMATION
# ==================================================

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Algorithm", "XGBoost")

with info2:
    st.metric("R² Score", "0.9411")

with info3:
    st.metric("MAE", "₹93,254")

st.caption(
    "Prediction is an estimate generated by the trained machine-learning model. "
    "Actual market prices may vary."
)
