import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

if st.button("🔮 Predict Customer Churn"):

    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["churn_prediction"]
            probability = result["churn_probability"]

            st.subheader("Prediction Result")

            if prediction == "Yes":
                st.error("⚠️ Customer is likely to churn.")
            else:
                st.success("✅ Customer is unlikely to churn.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Prediction",
                    prediction
                )

            with col2:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(probability)

        else:

            st.error(
                f"API returned status code {response.status_code}"
            )

            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            f"Could not connect to the FastAPI backend at {API_URL}"
        )

        st.info(
            "Make sure the FastAPI backend is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out."
        )

    except Exception as error:

        st.error(
            f"Unexpected error: {error}"
        )

st.divider()

st.caption(
    f"FastAPI Backend: {API_URL}"
)
