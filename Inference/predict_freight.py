import joblib
import pandas as pd
import os
import numpy as np

# 1. Update the filename here to match what is actually in your models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "predict_freight_cost.pkl")

def load_model(model_path: str = MODEL_PATH):
    """Load the trained regression model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def predict_freight_cost(input_data):
    """Predict freight cost using 6 features."""
    model = load_model()
    input_df = pd.DataFrame(input_data)
    
    # Standard features used during training
    feature_cols = [
        'invoice_dollars', 
        'days_po_to_invoice', 
        'days_to_pay', 
        'total_brands', 
        'total_item_quantity', 
        'total_item_dollars'
    ]
    
    # Ensure columns exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0 
            
    # Predict using .values to avoid feature name warnings
    X = input_df[feature_cols].values
    input_df['Predicted_Freight_Amt'] = model.predict(X).round(2)
    
    return input_df

if __name__ == "__main__":
    sample_data = {
    "PONumber": [101, 102],
    "invoice_dollars": [25000, 500],        # Huge difference
    "days_po_to_invoice": [5, 2],
    "days_to_pay": [30, 10],
    "total_brands": [5, 1],                # Different
    "total_item_quantity": [50, 1],         # Different
    "total_item_dollars": [24000, 450]      # Different
}

    
    try:
        results = predict_freight_cost(sample_data)
        print("\n--- Freight Cost Predictions ---")
        print(results[['PONumber', 'invoice_dollars', 'Predicted_Freight_Amt']])
    except Exception as e:
        print(f"Prediction failed: {e}")
