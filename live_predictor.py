import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import time
import os
import requests
import re
import matplotlib.pyplot as plt

# --- NEW: Plotting Functions ---

def setup_plots():
    """
    Initializes and displays two plot windows for live data.
    Enables interactive mode for real-time updates.
    """
    plt.ion() # Enable interactive mode
    
    # Figure 1: Win Probability
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title('Live Win Probability')
    ax1.set_xlabel('Play Number')
    ax1.set_ylabel('Home Team Win Probability')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Figure 2: Score Trend
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title('Live Score Trend')
    ax2.set_xlabel('Play Number')
    ax2.set_ylabel('Points')
    ax2.grid(True)

    plt.show()
    return fig1, ax1, fig2, ax2

def update_plots(fig1, ax1, fig2, ax2, prob_history, home_score_history, away_score_history, home_team, away_team):
    """
    Clears and redraws the plots with the latest data.
    """
    # --- Update Win Probability Plot ---
    ax1.clear()
    ax1.set_title(f'Live Win Probability: {away_team} @ {home_team}')
    ax1.set_xlabel('Play Number')
    ax1.set_ylabel('Home Team Win Probability')
    ax1.plot(prob_history, label=f'{home_team} Win Prob', color='blue')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='50%')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(left=0, right=max(1, len(prob_history) - 1))
    ax1.legend(loc='upper left')
    ax1.grid(True)
    fig1.canvas.draw()
    
    # --- Update Score Trend Plot ---
    ax2.clear()
    ax2.set_title(f'Live Score Trend: {away_team} @ {home_team}')
    ax2.set_xlabel('Play Number')
    ax2.set_ylabel('Points')
    ax2.plot(home_score_history, label=f'{home_team} Score', color='green')
    ax2.plot(away_score_history, label=f'{away_team} Score', color='red')
    ax2.set_xlim(left=0, right=max(1, len(home_score_history) - 1))
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig2.canvas.draw()
    
    plt.pause(0.1) # Pause to allow plots to update


# --- MODIFIED: More robust time parsing function ---
def parse_time_remaining(time_str, period):
    """
    Parses time remaining from various string formats (e.g., 'PT12M00.0S' or '12:00').
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0
    
    # Handle live data format, e.g., 'PT12M00.0S'
    if time_str.startswith('PT'):
        try:
            mins_match = re.search(r'(\d+)M', time_str)
            secs_match = re.search(r'(\d+)\.?\d*S', time_str)
            mins = int(mins_match.group(1)) if mins_match else 0
            secs = int(secs_match.group(1)) if secs_match else 0
        except (AttributeError, ValueError):
            return 0
    # Handle historical/simple format, e.g., '12:00'
    elif ':' in time_str:
        try:
            mins, secs = map(int, time_str.split(':'))
        except ValueError:
            return 0
    else:
        return 0

    seconds_in_period = mins * 60 + secs
    
    if period <= 4:
        return seconds_in_period + (4 - period) * 720
    else: # Overtime periods are 5 minutes (300 seconds)
        return seconds_in_period

# --- Game Finding Function (No changes) ---
def find_games_on_date(date_str, find_finished_games=False):
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
        data_frames = board.get_data_frames()
        games = data_frames[0]
        line_score = data_frames[1]
        
        team_abbrevs = line_score[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
        merged_df = pd.merge(games, team_abbrevs, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='inner')
        merged_df.rename(columns={'TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}, inplace=True)
        merged_df.drop(columns='TEAM_ID', inplace=True)
        final_merged_df = pd.merge(merged_df, team_abbrevs, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='inner')
        final_merged_df.rename(columns={'TEAM_ABBREVIATION': 'VISITOR_TEAM_ABBREVIATION'}, inplace=True)
        final_merged_df.drop(columns='TEAM_ID', inplace=True)

        if find_finished_games:
            return final_merged_df[final_merged_df['GAME_STATUS_TEXT'].str.contains('Final')]
        else:
            return final_merged_df[~final_merged_df['GAME_STATUS_TEXT'].str.contains('Final')]
    except Exception as e:
        print(f"Error fetching scoreboard for {date_str}: {e}")
        return pd.DataFrame()

def run_live_predictor():
    try:
        model = joblib.load('nba_win_predictor_xgb.joblib')
        print("Successfully loaded 'nba_win_predictor_xgb.joblib'")
    except FileNotFoundError:
        print("Error: Model file not found. Please run a training script first.")
        return

    while True:
        today_str = datetime.now().strftime('%Y-%m-%d')
        games_today = find_games_on_date(today_str)

        if games_today.empty:
            print(f"No schedulable NBA games for today ({today_str}).")
            fallback_date = '2024-04-10'
            print(f"Falling back to a sample date for demonstration: {fallback_date}")
            games_today = find_games_on_date(fallback_date, find_finished_games=True)
            
            if games_today.empty:
                print("Could not fetch fallback games either. Checking again in 5 minutes.")
                time.sleep(300)
                continue

        print("\n--- Available Games ---")
        for index, game in games_today.iterrows():
            print(f"{index}: {game['GAMECODE']} - {game['GAME_STATUS_TEXT']}")
        
        try:
            choice = int(input("Select a game index to track (or press Ctrl+C to exit): "))
            selected_game = games_today.loc[choice]
        except (ValueError, KeyError):
            print("Invalid choice. Please try again.")
            continue

        game_id = selected_game['GAME_ID']
        home_team_name = selected_game['HOME_TEAM_ABBREVIATION']
        away_team_name = selected_game['VISITOR_TEAM_ABBREVIATION']
        
        print(f"\nPreparing to track game: {away_team_name} @ {home_team_name}")
        
        fig1, ax1, fig2, ax2 = setup_plots()
        game_is_live = True
        last_known_play_count = 0
        
        while game_is_live:
            try:
                live_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
                response = requests.get(live_url, timeout=30)
                response.raise_for_status()
                live_data = response.json()
                
                plays = live_data.get('game', {}).get('actions', [])
                
                if not plays:
                    game_status = selected_game['GAME_STATUS_TEXT']
                    if "PM" in game_status or "AM" in game_status:
                        print(f"Game is scheduled for {game_status}. Waiting for tip-off... (Checking again in 15s)")
                    else:
                        print(f"Game is live. Waiting for first play to be logged by the API... (Checking again in 15s)")
                    time.sleep(15)
                    continue
                
                df = pd.DataFrame(plays)

                if len(df) > last_known_play_count:
                    last_known_play_count = len(df)
                    
                    # --- FIX 1: Update to modern .ffill() method to remove warning ---
                    df['HOME_SCORE'] = pd.to_numeric(df['scoreHome']).ffill().fillna(0).astype(int)
                    df['AWAY_SCORE'] = pd.to_numeric(df['scoreAway']).ffill().fillna(0).astype(int)
                    
                    # --- FIX 2: Rename 'period' column to 'PERIOD' to match model's expectation ---
                    df.rename(columns={'period': 'PERIOD'}, inplace=True)
                    
                    home_score_history = df['HOME_SCORE'].tolist()
                    away_score_history = df['AWAY_SCORE'].tolist()
                    
                    # Engineer features for the entire DataFrame to get full history
                    df['SCORE_MARGIN'] = df['HOME_SCORE'] - df['AWAY_SCORE']
                    df['SECONDS_REMAINING'] = df.apply(lambda row: parse_time_remaining(row['clock'], row['PERIOD']), axis=1)
                    df['SCORE_MARGIN_SQ'] = df['SCORE_MARGIN'] ** 2
                    df['SECONDS_REMAINING_SQ'] = df['SECONDS_REMAINING'] ** 2
                    
                    # Get predictions for all plays
                    features_for_model = df[['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']]
                    all_probas = model.predict_proba(features_for_model)
                    prob_history = all_probas[:, 1].tolist()

                    update_plots(fig1, ax1, fig2, ax2, prob_history, home_score_history, away_score_history, home_team_name, away_team_name)

                    last_play = df.iloc[-1]
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"--- Live Tracking ---")
                    print(f"Game: {away_team_name} @ {home_team_name} | Total Plays: {len(df)}")
                    print(f"Score: {away_team_name} {last_play['AWAY_SCORE']} - {home_team_name} {last_play['HOME_SCORE']}")
                    print(f"Time:  {last_play['clock']} - Period {last_play['PERIOD']}") # Use renamed column
                    print("\n--- Win Probability ---")
                    print(f"  {home_team_name} (Home): {prob_history[-1]:.2%}")
                    print(f"  {away_team_name} (Away): {1-prob_history[-1]:.2%}")
                    
                    is_game_over = last_play.get('actionType') == 'game' and last_play.get('subType') == 'end'
                    if is_game_over:
                         print("\n--- GAME OVER ---")
                         game_is_live = False
                         input("Press Enter to close plots and track another game...")
                         continue

                time.sleep(5)

            except KeyboardInterrupt:
                game_is_live = False
                print("\nStopping tracker.")
            except Exception as e:
                print(f"An error occurred: {e}. Retrying in 30 seconds.")
                time.sleep(30)
        
        plt.close('all') # Close plot windows when a game finishes

if __name__ == '__main__':
    run_live_predictor()