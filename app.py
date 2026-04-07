import streamlit as st
import pandas as pd
import numpy as np

# Ensure your 'inference' folder has an __init__.py file
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------------------
# Header Section
# --------------------------------------------------------------

st.markdown("""
# Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------

st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- Improved cost forecasting
- Reduced invoice fraud & anomalies
- Faster finance operations
""")

# --------------------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------------------

if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice based on invoice details
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):                        # ← everything INSIDE form
        col1, col2 = st.columns(2)

        with col1:
            invoice_dollars     = st.number_input("Invoice Dollars",    min_value=1.0, value=25000.0)
            days_po_to_invoice  = st.number_input("Days PO to Invoice", min_value=0,   value=5)
            days_to_pay         = st.number_input("Days to Pay",        min_value=0,   value=30)

        with col2:
            total_brands        = st.number_input("Total Brands",        min_value=1, value=5)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=50)
            total_item_dollars  = st.number_input("Total Item Dollars",  min_value=1.0, value=24000.0)

        submit_freight = st.form_submit_button("Predict Freight Cost")  # ← inside form

    if submit_freight:
        input_data = {
            "invoice_dollars":    [invoice_dollars],
            "days_po_to_invoice": [days_po_to_invoice],
            "days_to_pay":        [days_to_pay],
            "total_brands":       [total_brands],
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        prediction_result = predict_freight_cost(input_data)
        prediction_value  = prediction_result['Predicted_Freight_Amt'].iloc[0]

        st.success("Prediction completed successfully.")
        st.metric(label="Estimated Freight Cost", value=f"${float(prediction_value):,.2f}")

# --------------------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------------------

else:
    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
        Predict whether a vendor invoice should be **flagged for manual approval**
        based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):                   # ← everything INSIDE form
        col1, col2, col3 = st.columns(3)

        with col1:
            freight_input      = st.number_input("Freight Cost",        min_value=0.0, value=1.73)
            days_po_to_invoice = st.number_input("Days PO to Invoice",  min_value=0,   value=5)

        with col2:
            invoice_dollars    = st.number_input("Invoice Dollars",     min_value=1.0, value=352.95)
            total_item_quantity= st.number_input("Total Item Quantity", min_value=1,   value=162)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars",  min_value=1.0, value=2476.0)
            days_to_pay        = st.number_input("Days to Pay",         min_value=0,   value=30)

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")    # ← inside form

    if submit_flag:
        input_data = {
            "invoice_dollars":    [invoice_dollars],
            "Freight":            [freight_input],
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "days_po_to_invoice": [days_po_to_invoice],
            "days_to_pay":        [days_to_pay]
        }

        flag_result    = predict_invoice_flag(input_data)
        flag_prediction = flag_result['Predicted_Flag']
        is_flagged     = bool(flag_prediction.iloc[0])

        if is_flagged:
            st.error("⚠️ Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("✅ Invoice is **SAFE for Auto-Approval**")