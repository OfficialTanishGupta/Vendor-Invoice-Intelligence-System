import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    """Load trained classifier model."""
    return joblib.load(model_path)

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    input_data: dict or list of dicts
    """
    model = load_model()
    
    # Ensures a single dict is treated as one row
    if isinstance(input_data, dict):
        input_data = [input_data]
        
    input_df = pd.DataFrame(input_data)
    
    # Predict and add column
    # Use .iloc[0] or similar if you only want the value, 
    # but returning the DF works with your app.py logic
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

# FIX: Added 'pass' so the block isn't empty
if __name__ == "__main__":
    pass 
