import joblib
from pathlib import Path
# 1. ADD THESE IMPORTS (Adjust the filenames if they are different)
# Change load_vendor_invoice_data -> load_invoice_data
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    # 2. DEFINE db_path so the function knows where to look
    db_path = r"C:\Users\Tanish_Gupta\OneDrive\Desktop\ML Projects\Invoice Intelligence System\Freight_cost_prediction\data\inventory.db"
    
    # Ensure the models directory exists
    Path("models").mkdir(exist_ok=True)

    # load data
    df = load_invoice_data() 
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'models/scaler.pkl'
    )

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')
    
if __name__ == '__main__':
    main()
