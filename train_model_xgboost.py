import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def train_xgboost_model():
    """
    Loads the collected game data, performs hyperparameter tuning on an
    XGBoost model, evaluates it, and saves the best model.
    """
    print("Loading game data...")
    try:
        df = pd.read_csv('data/nba_game_data.csv')
    except FileNotFoundError:
        print("Error: 'nba_game_data.csv' not found. Please run the data collection script first.")
        return

    print(f"Data loaded successfully. Shape: {df.shape}")

    # --- Feature Engineering ---
    # Create the same features used for the other models
    print("Engineering new features...")
    df['SCORE_MARGIN'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    df['SCORE_MARGIN_SQ'] = df['SCORE_MARGIN'] ** 2
    df['SECONDS_REMAINING_SQ'] = df['SECONDS_REMAINING'] ** 2
    
    # Define features (X) and target (y)
    features = ['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']
    target = 'HOME_TEAM_WINS'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into training and testing sets (Train: {len(X_train)}, Test: {len(X_test)}).")

    # --- Hyperparameter Tuning with GridSearchCV for XGBoost ---
    print("\nStarting XGBoost hyperparameter tuning...")

    # 1. Define the XGBoost model
    # Use 'eval_metric' to avoid a deprecation warning
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # 2. Define the parameter grid
    # This is a focused grid. You can expand this for a more exhaustive search.
    param_grid = {
        'n_estimators': [100, 200],         # Number of trees
        'max_depth': [3, 5],              # Maximum depth of a tree
        'learning_rate': [0.1, 0.2]       # Step size shrinkage
    }

    # 3. Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Using 3-fold CV as XGBoost can be slower
        scoring='neg_log_loss',
        verbose=1,
        n_jobs=-1
    )

    # 4. Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # 5. Get the best model and its parameters
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation log loss: {-grid_search.best_score_:.4f}")
    # --- End of Hyperparameter Tuning Section ---

    print("\nEvaluating the best XGBoost model on the test set...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int) # Get binary predictions from probabilities

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n--- XGBoost Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("--------------------------------")

    # Save the trained model to a file
    model_filename = 'nba_win_predictor_xgb.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nBest XGBoost model saved to '{model_filename}'")

if __name__ == '__main__':
    train_xgboost_model()