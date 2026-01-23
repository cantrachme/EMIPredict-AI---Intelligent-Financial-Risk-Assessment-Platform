import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# =====================================================
# App Config
# =====================================================
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
)


# =====================================================
# Load Models
# =====================================================
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models")

    with open(os.path.join(model_dir, "emi_eligibility_model.pkl"), "rb") as f:
        eligibility_model = pickle.load(f)

    with open(os.path.join(model_dir, "max_emi_model.pkl"), "rb") as f:
        emi_model = pickle.load(f)

    return eligibility_model, emi_model


eligibility_model, emi_model = load_models()

# =====================================================
# Sidebar
# =====================================================
st.sidebar.title("🏦 EMIPredict AI")
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
# Helpers
# =====================================================
def predict_eligibility(df):
    pred = eligibility_model.predict(df)[0]
    return "Approved" if pred == 1 else "Rejected"


def predict_max_emi(df):
    return float(emi_model.predict(df)[0])


# =====================================================
# HOME
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
# EMI ELIGIBILITY
# =====================================================
elif page == "✅ EMI Eligibility Check":
    st.title("✅ EMI Eligibility Check")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
        marital_status = st.selectbox("Marital Status", [0, 1])
        education = st.selectbox("Education", [0, 1, 2])

        monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 500000, 75000)
        employment_type = st.selectbox("Employment Type", [0, 1])
        years_of_employment = st.slider("Years of Employment", 0, 40, 5)
        company_type = st.selectbox("Company Type", [0, 1])

    with col2:
        house_type = st.selectbox("House Type", [0, 1])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
        family_size = st.slider("Family Size", 1, 10, 4)
        dependents = st.slider("Dependents", 0, 5, 1)

        school_fees = st.number_input("School Fees (₹)", 0, 50000, 5000)
        college_fees = st.number_input("College Fees (₹)", 0, 50000, 0)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 30000, 5000)
        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)", 0, 50000, 12000
        )

    with col3:
        other_monthly_expenses = st.number_input("Other Expenses (₹)", 0, 50000, 5000)
        existing_loans = st.selectbox("Existing Loans", [0, 1])
        current_emi_amount = st.number_input("Current EMI (₹)", 0, 100000, 5000)

        credit_score = st.slider("Credit Score", 300, 900, 750)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 10000000, 200000)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 100000)

        emi_scenario = st.selectbox("EMI Scenario", [0, 1, 2])
        requested_amount = st.number_input(
            "Requested Loan Amount (₹)", 10000, 5000000, 500000
        )
        requested_tenure = st.slider("Requested Tenure (Years)", 1, 30, 10)

    if st.button("Check Eligibility"):
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "gender": gender,
                    "marital_status": marital_status,
                    "education": education,
                    "monthly_salary": monthly_salary,
                    "employment_type": employment_type,
                    "years_of_employment": years_of_employment,
                    "company_type": company_type,
                    "house_type": house_type,
                    "monthly_rent": monthly_rent,
                    "family_size": family_size,
                    "dependents": dependents,
                    "school_fees": school_fees,
                    "college_fees": college_fees,
                    "travel_expenses": travel_expenses,
                    "groceries_utilities": groceries_utilities,
                    "other_monthly_expenses": other_monthly_expenses,
                    "existing_loans": existing_loans,
                    "current_emi_amount": current_emi_amount,
                    "credit_score": credit_score,
                    "bank_balance": bank_balance,
                    "emergency_fund": emergency_fund,
                    "emi_scenario": emi_scenario,
                    "requested_amount": requested_amount,
                    "requested_tenure": requested_tenure,
                }
            ]
        )

        result = predict_eligibility(input_df)

        if result == "Approved":
            st.success("🎉 EMI Eligibility: APPROVED")
        else:
            st.error("❌ EMI Eligibility: REJECTED")

# =====================================================
# MAX EMI
# =====================================================
elif page == "💸 Max EMI Prediction":
    st.title("💸 Maximum EMI Prediction")

    col1, col2 = st.columns(2)

    with col1:
        monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 500000, 75000)
        credit_score = st.slider("Credit Score", 300, 900, 750)
        current_emi_amount = st.number_input("Current EMI (₹)", 0, 100000, 5000)

    with col2:
        bank_balance = st.number_input("Bank Balance (₹)", 0, 10000000, 200000)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 100000)
        requested_amount = st.number_input(
            "Requested Loan Amount (₹)", 10000, 5000000, 500000
        )
        requested_tenure = st.slider("Requested Tenure (Years)", 1, 30, 10)

    if st.button("Predict Max EMI"):
        # Step 1: Build FULL feature vector
        full_df = pd.DataFrame(
            [
                {
                    "age": 30,
                    "gender": 1,
                    "marital_status": 1,
                    "education": 2,
                    "monthly_salary": monthly_salary,
                    "employment_type": 1,
                    "years_of_employment": 5,
                    "company_type": 1,
                    "house_type": 1,
                    "monthly_rent": 10000,
                    "family_size": 4,
                    "dependents": 1,
                    "school_fees": 5000,
                    "college_fees": 0,
                    "travel_expenses": 5000,
                    "groceries_utilities": 12000,
                    "other_monthly_expenses": 5000,
                    "existing_loans": 1,
                    "current_emi_amount": current_emi_amount,
                    "credit_score": credit_score,
                    "bank_balance": bank_balance,
                    "emergency_fund": emergency_fund,
                    "emi_scenario": 1,
                    "requested_amount": requested_amount,
                    "requested_tenure": requested_tenure,
                }
            ]
        )

        # Step 2: Predict eligibility FIRST
        eligibility = eligibility_model.predict(full_df)[0]

        # Step 3: Inject eligibility into features
        full_df["emi_eligibility"] = eligibility

        # Step 4: Predict Max EMI
        max_emi = predict_max_emi(full_df)

        st.metric("💰 Maximum Affordable EMI", f"₹ {max_emi:,.0f}")


# =====================================================
# DATA INSIGHTS (UNCHANGED)
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
