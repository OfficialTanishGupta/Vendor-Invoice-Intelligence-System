from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
import warnings

# Silence the specific Python 3.14 compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_random_forest(X_train, y_train):
    # Set n_jobs to 1 or 2 to avoid the Windows/Python 3.14 parallel processing bug
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1 
    )

    # Simplified grid (from 216 combinations down to 18)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,        # Reduced from 5 to 3 for much faster training
        n_jobs=1,    # Set to 1 to stop the "delayed" warnings
        verbose=1    # Set to 1 so you can see the progress
    )

    print("Training models... please wait.")
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    # Added zero_division=0 to prevent errors if a class isn't predicted
    report = classification_report(y_test, preds, zero_division=0)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
