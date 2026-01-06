import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# =====================================================
# App Configuration
# =====================================================
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
)


# =====================================================
# Load Pickle Models
# =====================================================
@st.cache_resource
def load_models():
    try:
        base_dir = os.path.dirname(__file__)
        model_dir = os.path.join(base_dir, "models")

        with open(os.path.join(model_dir, "emi_eligibility_model.pkl"), "rb") as f:
            eligibility_model = pickle.load(f)

        with open(os.path.join(model_dir, "max_emi_model.pkl"), "rb") as f:
            emi_model = pickle.load(f)

        return eligibility_model, emi_model

    except Exception as e:
        st.error("❌ Failed to load models")
        st.exception(e)
        st.stop()


eligibility_model, emi_model = load_models()

# =====================================================
# Sidebar Navigation
# =====================================================
st.sidebar.title("🏦 EMIPredict AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "✅ EMI Eligibility Check",
        "💸 Max EMI Prediction",
        "📊 Data Insights",
    ],
)


# =====================================================
# Helper Functions
# =====================================================
def predict_eligibility(input_df):
    prediction = eligibility_model.predict(input_df)[0]
    return "Approved" if prediction == 1 else "Rejected"


def predict_max_emi(input_df):
    return float(emi_model.predict(input_df)[0])


# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.title("💰 EMIPredict AI")
    st.subheader("Intelligent Financial Risk Assessment Platform")

    st.markdown(
        """
        **EMIPredict AI** is an end-to-end AI system that helps banks and
        financial institutions make **accurate loan decisions**.

        ### 🚀 Key Features
        ✔ EMI Eligibility Classification  
        ✔ Maximum EMI Prediction  
        ✔ ML-driven Risk Assessment  
        ✔ Interactive Data Insights  

        ### 🛠 Tech Stack
        - **Machine Learning:** Scikit-learn, XGBoost  
        - **Model Management:** MLflow (Training Phase)  
        - **Deployment:** Streamlit Cloud  
        - **Frontend:** Streamlit + Plotly  
        """
    )

# =====================================================
# EMI ELIGIBILITY PAGE
# =====================================================
elif page == "✅ EMI Eligibility Check":
    st.title("✅ EMI Eligibility Check")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        monthly_income = st.number_input("Monthly Income (₹)", 0, 500000, 50000)
        monthly_expenses = st.number_input("Monthly Expenses (₹)", 0, 500000, 20000)

    with col2:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        existing_emi = st.number_input("Existing EMI (₹)", 0, 500000, 0)

    if st.button("Check Eligibility"):
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "monthly_income": monthly_income,
                    "monthly_expenses": monthly_expenses,
                    "credit_score": credit_score,
                    "existing_emi": existing_emi,
                }
            ]
        )

        result = predict_eligibility(input_df)

        if result == "Approved":
            st.success("🎉 EMI Eligibility: **APPROVED**")
        else:
            st.error("❌ EMI Eligibility: **REJECTED**")

# =====================================================
# MAX EMI PREDICTION PAGE
# =====================================================
elif page == "💸 Max EMI Prediction":
    st.title("💸 Maximum EMI Prediction")

    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", 0, 500000, 50000)
        monthly_expenses = st.number_input("Monthly Expenses (₹)", 0, 500000, 20000)

    with col2:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        existing_emi = st.number_input("Existing EMI (₹)", 0, 500000, 0)

    if st.button("Predict Max EMI"):
        input_df = pd.DataFrame(
            [
                {
                    "monthly_income": monthly_income,
                    "monthly_expenses": monthly_expenses,
                    "credit_score": credit_score,
                    "existing_emi": existing_emi,
                }
            ]
        )

        max_emi = predict_max_emi(input_df)

        st.metric("💰 Maximum Affordable EMI", f"₹ {max_emi:,.0f}")

# =====================================================
# DATA INSIGHTS PAGE
# =====================================================
elif page == "📊 Data Insights":
    st.title("📊 Financial Insights & Risk Trends")

    # ----------------- Line Chart -----------------
    df = pd.DataFrame(
        {
            "Monthly Income": [30000, 50000, 70000, 90000, 120000],
            "Approval Rate": [0.35, 0.55, 0.75, 0.85, 0.93],
        }
    )

    fig = px.line(
        df,
        x="Monthly Income",
        y="Approval Rate",
        markers=True,
        title="Income vs EMI Approval Rate",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----------------- Image Visuals -----------------
    st.subheader("📈 EMI Eligibility & Risk Analysis")

    base_dir = os.path.dirname(__file__)
    viz_dir = os.path.join(base_dir, "Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            os.path.join(viz_dir, "EMI Eligibility Distribution.png"),
            caption="EMI Eligibility Distribution",
            use_container_width=True,
        )

        st.image(
            os.path.join(viz_dir, "EMI Eligibility Across Scenarios.png"),
            caption="EMI Eligibility Across Scenarios",
            use_container_width=True,
        )
