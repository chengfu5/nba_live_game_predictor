from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests # Import the requests library to fetch data

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('nba_win_predictor_xgb.joblib')
    print("Successfully loaded 'nba_win_predictor_xgb.joblib'")
except FileNotFoundError:
    print("FATAL ERROR: 'nba_win_predictor_xgb.joblib' not found.")
    model = None

# --- NEW: Endpoint to fetch scoreboard data ---
@app.route('/scoreboard/<date_str>', methods=['GET'])
def get_scoreboard(date_str):
    """
    Fetches scoreboard data from the NBA's CDN for a given date.
    This acts as a proxy to bypass browser CORS issues.
    """
    try:
        url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{date_str}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (like 404)
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching scoreboard from NBA: {e}")
        # Return a structured error that the front-end can understand
        return jsonify({"error": "Failed to fetch scoreboard data from NBA API."}), 500

# --- NEW: Endpoint to fetch play-by-play data ---
@app.route('/pbp/<game_id>', methods=['GET'])
def get_pbp(game_id):
    """
    Fetches play-by-play data for a specific game ID.
    Acts as a proxy to bypass browser CORS issues.
    """
    try:
        url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PBP data from NBA: {e}")
        return jsonify({"error": f"Failed to fetch PBP data for game {game_id}."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives live game data, engineers features, and returns a win probability.
    """
    if not model:
        return jsonify({'error': 'Model is not loaded on the server.'}), 500
    try:
        data = request.get_json()
        score_margin = data['score_margin']
        seconds_remaining = data['seconds_remaining']
        period = data['period']
        
        score_margin_sq = score_margin ** 2
        seconds_remaining_sq = seconds_remaining ** 2

        features_df = pd.DataFrame(
            [[score_margin, seconds_remaining, period, score_margin_sq, seconds_remaining_sq]],
            columns=['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']
        )
        
        win_proba = model.predict_proba(features_df)
        home_win_probability = win_proba[0][1]

        # --- FIX: Convert the numpy.float32 to a native Python float ---
        # This ensures the data type is JSON serializable.
        return jsonify({'home_win_probability': float(home_win_probability)})
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'Could not process the prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)