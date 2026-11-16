
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load the trained ML pipeline
MODEL_PATH = BASE_DIR / "models" / "return_model.pkl"

model = joblib.load(MODEL_PATH)


# Load dropdown options
OPTIONS_PATH = BASE_DIR / "models" / "input_options.pkl"

input_options = joblib.load(OPTIONS_PATH)


# Streamlit page configuration
st.set_page_config(
    page_title="ReturnRadar",
    page_icon="📦",
    layout="wide"
)

# Application heading
st.title("ReturnRadar")
st.subheader("E-Commerce Return Risk Prediction")

st.write(
    "Predict the likelihood of an e-commerce order being returned "
    "using machine learning."
)

st.success("Machine learning model loaded successfully!")


st.divider()

st.header("Predict Return Risk")

st.write("Enter the order and customer details below.")

with st.form("return_prediction_form"):

    st.subheader("Product Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        item_id = st.selectbox(
            "Item ID",
            options=input_options["item_id"]
        )

    with col2:
        brand_id = st.selectbox(
            "Brand ID",
            options=input_options["brand_id"]
        )

    with col3:
        item_price = st.number_input(
            "Item Price",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        item_size = st.selectbox(
            "Item Size",
            options=input_options["item_size"]
        )

    with col5:
        item_color = st.selectbox(
            "Item Color",
            options=input_options["item_color"]
        )

    with col6:
        user_title = st.selectbox(
            "Customer Title",
            options=input_options["user_title"]
        )

    st.subheader("Customer and Delivery Details")

    col7, col8, col9 = st.columns(3)

    with col7:
        user_state = st.selectbox(
            "Customer State",
            options=input_options["user_state"]
        )

    with col8:
        age = st.number_input(
            "Customer Age",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

    with col9:
        delivery_missing = st.selectbox(
            "Delivery Information Missing?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    col10, col11, col12 = st.columns(3)

    with col10:
        delivery_days = st.number_input(
            "Delivery Days",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

    with col11:
        order_month = st.selectbox(
            "Order Month",
            options=list(range(1, 13)),
            index=5
        )

    with col12:
        order_day_of_week = st.selectbox(
            "Order Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )

    days_since_registration = st.number_input(
        "Days Since Customer Registration",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "Predict Return Risk",
        use_container_width=True
    )

if submitted:

    # Create a DataFrame with the same 13 features
    input_data = pd.DataFrame([{
        "item_id": item_id,
        "item_size": item_size,
        "item_color": item_color,
        "brand_id": brand_id,
        "item_price": item_price,
        "user_title": user_title,
        "user_state": user_state,
        "age": age,
        "delivery_days": delivery_days,
        "delivery_missing": delivery_missing,
        "order_month": order_month,
        "order_day_of_week": order_day_of_week,
        "days_since_registration": days_since_registration
    }])

    try:
        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Calculate return probability
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.header("Prediction Result")

        if prediction == 1:
            st.error("This order is predicted to be RETURNED.")
        else:
            st.success("This order is predicted NOT TO BE RETURNED.")

        st.metric(
            label="Return Probability",
            value=f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

    except Exception as e:
        st.error(f"Unable to generate prediction: {e}")

