import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib
import warnings

def train():
    """
    Loads the collected data, trains a logistic regression model,
    and saves it to a file.
    """
    print("Loading data...")
    try:
        df = pd.read_csv('data/nba_game_data.csv')
    except FileNotFoundError:
        print("Error: nba_game_data.csv not found.")
        print("Please run collect_data.py first to generate the dataset.")
        return

    print("Preparing data for training...")
    # Drop rows with missing values that might have occurred during collection
    df.dropna(subset=['HOME_SCORE', 'AWAY_SCORE', 'SECONDS_REMAINING'], inplace=True)

    # Feature Engineering: Score margin is a powerful predictor
    df['SCORE_MARGIN'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    df['SCORE_MARGIN_SQ'] = df['SCORE_MARGIN'] ** 2
    df['SECONDS_REMAINING_SQ'] = df['SECONDS_REMAINING'] ** 2

    # Select features (X) and target variable (y)
    features = ['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']
    target = 'HOME_TEAM_WINS'

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training model on {len(X_train)} data points...")
    # --- NEW: Hyperparameter Tuning with GridSearchCV ---
    print("\nStarting hyperparameter tuning with GridSearchCV...")

    # 1. Define the model
    model = LogisticRegression(solver='liblinear', random_state=42)

    # 2. Define the parameter grid to search
    # These 'C' values represent different regularization strengths
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    }

    # 3. Set up GridSearchCV
    # cv=5 means 5-fold cross-validation
    # scoring='neg_log_loss' is a good metric for probability models
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_log_loss', 
        verbose=1, # This will print progress updates
        n_jobs=-1  # Use all available CPU cores
    )

    # 4. Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # 5. Get the best model and its parameters
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation log loss: {-grid_search.best_score_:.4f}")
    # --- End of Hyperparameter Tuning Section ---

    print("\nEvaluating the best model on the test set...")
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("--------------------------")

    # Save the trained model to a file
    model_filename = 'nba_win_predictor.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nBest model saved to '{model_filename}'")

if __name__ == '__main__':
    train()
