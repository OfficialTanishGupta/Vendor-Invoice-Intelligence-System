import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("..", "models", "predict_flag_invoice.pkl")
def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    # Using 'with' is good practice for loading files
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    """
    model = load_model()
    
    # 2. Fixed typo: pd.dataFrame -> pd.DataFrame (Capital 'D')
    input_df = pd.DataFrame(input_data)
    
    # 3. Note: The model expects columns like 'invoice_dollars', 
    # so we match the training features here.
    input_df['Predicted_freight'] = model.predict(input_df.values).round(2)
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    # 4. Note: I updated "Dollars" to "invoice_dollars" to match your training FEATURES
    sample_data = {
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [100, 50, 20, 5],
        "total_item_quantity": [10, 5, 2, 1],
        "total_item_dollars": [18000, 8500, 2900, 190],
        "days_po_to_invoice": [5, 5, 5, 5],
        "days_to_pay": [30, 30, 30, 30]
    }
    
    prediction = predict_freight_cost(sample_data)
    print(prediction)
