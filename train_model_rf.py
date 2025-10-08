import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

def train_random_forest_model():
    """
    Loads the collected game data, engineers features, performs hyperparameter
    tuning for a RandomForestClassifier, trains the best model, evaluates it, 
    and saves it.
    """
    print("Loading game data...")
    try:
        df = pd.read_csv('data/nba_game_data.csv')
    except FileNotFoundError:
        print("Error: 'nba_game_data.csv' not found. Please run the data collection script first.")
        return

    print("Engineering features to match simulation script...")
    df['SCORE_MARGIN'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    df['SCORE_MARGIN_SQ'] = df['SCORE_MARGIN'] ** 2
    df['SECONDS_REMAINING_SQ'] = df['SECONDS_REMAINING'] ** 2
    
    # Define features (X) and target (y)
    features = [
        'SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD',
        'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ'
    ]
    target = 'HOME_TEAM_WINS'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")

    # --- Hyperparameter Tuning for Random Forest ---
    print("\nStarting hyperparameter tuning for Random Forest...")

    # 1. Define the model
    rf = RandomForestClassifier(random_state=42)

    # 2. Define the parameter grid to search
    # This is a smaller grid to keep tuning time reasonable.
    # You can expand this with more values if you have more time.
    param_grid = {
        'n_estimators': [100, 200],       # Number of trees in the forest
        'max_depth': [10, 20, None],      # Maximum depth of the tree
        'min_samples_leaf': [2, 4],       # Minimum number of samples required at a leaf node
        'min_samples_split': [5, 10]      # Minimum number of samples required to split a node
    }

    # 3. Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3,  # 3-fold cross-validation is faster for this complex model
        scoring='neg_log_loss', 
        verbose=2, # Print more detailed progress
        n_jobs=-1  # Use all available CPU cores
    )

    # 4. Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # 5. Get the best model and its parameters
    best_rf_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation log loss: {-grid_search.best_score_:.4f}")
    # --- End of Hyperparameter Tuning Section ---

    print("\nEvaluating the best Random Forest model on the test set...")
    y_pred = best_rf_model.predict(X_test)
    y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n--- Model Evaluation (Random Forest) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("------------------------------------------")

    # Save the trained model to a new file
    model_filename = 'nba_win_predictor_rf.joblib'
    joblib.dump(best_rf_model, model_filename)
    print(f"\nBest Random Forest model saved to '{model_filename}'")

if __name__ == '__main__':
    train_random_forest_model()
