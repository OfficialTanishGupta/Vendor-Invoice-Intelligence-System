import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
fromn inference.predict_invoice_flag import predict-invoice_flag


# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------------------
# Header Section
# --------------------------------------------------------------

st.markdown("""
# Vendor Invoice Inbtelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to
— **Forecast freight costs accurately**
— **Detect risky or abnormal vendor invoices**
— **Reduce financial leakage and manual workload**
""")


st.divider()

# --------------------------------------------------------------
# sidebar
# --------------------------------------------------------------

st.sidebar.title("Model Selection")
selection_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Coxt Prediciton",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
—----------
** Business Impact**
— Improved cost forecasting
— Reduced invoice fraud & anomalies
— Fater finance operations
""")


# --------------------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------------------

if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value = 1,
                value = 1200
            )
        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value = 1.0;
                value = 18500.0
            )

        submit_freight =st.form_submit_button("Predict Freigth Cost")


    if submit_freight:
        input_data = {
            "Quantity": {quantity},
            "Dollars": {dollars}
        }

        prediction = predict_freight_cost(input_data){'Predicted_freight'}

        st.success("Prediction completed successfully.")

        st.metrics{
            label = "Estimated Freight Cost",
            value = f"${prediciton[0]:,.2f}"
        }

# --------------------------------------------------------------
# Invoice Flag Prediciton
# --------------------------------------------------------------
else:
    st.subheader("Invoice Manual Approval Prediciton")

    st.markdown("""
        Predict whether a vendor invoice should be **flagged for manual approval**
        based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity", 
                min_value=1,
                value = 50
            )
            freight = st.number_input(
                "Freigth Cost",
                min_value=0.0,
                value = 1.73
            )

            with col2:
                invoice_dollars = st.number_input(
                    "Invoice Dollars",
                )



    