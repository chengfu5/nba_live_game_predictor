import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2, playbyplayv2
from datetime import datetime, timedelta
import time
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def parse_time_remaining(time_str, period):
    """
    Converts clock time and period into total seconds remaining in the game.
    """
    if pd.isna(time_str) or not isinstance(time_str, str) or ':' not in time_str:
        return 0
        
    try:
        mins, secs = map(int, time_str.split(':'))
    except ValueError:
        return 0

    seconds_in_period = mins * 60 + secs
    
    if period <= 4:
        full_periods_remaining = 4 - period
        return seconds_in_period + full_periods_remaining * 720
    else: # Overtime
        return seconds_in_period

def get_finished_games_from_board(board):
    """
    Helper function to process scoreboard dataframes into a clean list of finished games.
    This new logic is more robust and uses LineScore for team abbreviations.
    """
    data_frames = board.get_data_frames()
    games = data_frames[0]       # GameHeader
    line_score = data_frames[1]  # LineScore
    
    team_abbrevs = line_score[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()

    merged_df = pd.merge(
        games, 
        team_abbrevs, 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='inner'
    )
    merged_df.rename(columns={'TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}, inplace=True)
    merged_df.drop(columns='TEAM_ID', inplace=True)

    final_merged_df = pd.merge(
        merged_df,
        team_abbrevs,
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='inner'
    )
    final_merged_df.rename(columns={'TEAM_ABBREVIATION': 'AWAY_TEAM_ABBREVIATION'}, inplace=True)
    final_merged_df.drop(columns='TEAM_ID', inplace=True)
    
    finished_games = final_merged_df[final_merged_df['GAME_STATUS_TEXT'] == 'Final']
    return finished_games

def find_recent_finished_games():
    """Finds and returns a list of recently completed NBA games."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Searching for completed games from yesterday ({yesterday})...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=yesterday, timeout=60)
        return get_finished_games_from_board(board)
    except Exception as e:
        return pd.DataFrame()

def find_games_on_sample_date(date_str='2024-04-10'):
    """Finds completed games on a hardcoded date to ensure functionality during off-season."""
    print(f"\nNo games found for yesterday. Now searching on a sample date ({date_str}) from the last season...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=60)
        return get_finished_games_from_board(board)
    except Exception as e:
        print(f"Error fetching scoreboard for sample date {date_str}: {e}")
        return pd.DataFrame()

def plot_win_probability(probabilities, home_scores, away_scores, home_team, away_team):
    """
    Generates and displays a line plot of the win probabilities against the number of plays.
    """
    if not probabilities:
        print("No probability data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    # --- FIGURE 1: Win Probability ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # --- SUBPLOT 1: Win Probability ---
    ax1.plot(probabilities, label=f'{home_team} Win Probability', color='royalblue')
    ax1.set_title(f'Win Probability Chart', fontsize=14)
    ax1.set_ylabel('Home Team Win Probability', fontsize=12)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='50% (Toss-up)')
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True)

    # --- SUBPLOT 2: Score Progression ---
    ax2.plot(home_scores, color='green', linestyle='-', label=f'{home_team} Score')
    ax2.plot(away_scores, color='red', linestyle='-', label=f'{away_team} Score')
    ax2.set_title(f'Score Progression', fontsize=14)
    ax2.set_xlabel('Play Number', fontsize=12)
    ax2.set_ylabel('Game Score (Points)', fontsize=12)
    max_score = max(max(home_scores), max(away_scores))
    ax2.set_ylim(0, max_score + 10)
    ax2.set_xlim(0, len(home_scores) - 1)
    ax2.legend()
    ax2.grid(True)
    
    # Add a main title to the entire figure
    fig.suptitle(f'Game Summary: {away_team} @ {home_team}', fontsize=18, fontweight='bold')
    
    # Adjust layout to prevent titles from overlapping
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # rect leaves space for suptitle
    
    # Save the combined figure to a single file
    fig.savefig("figures/win_probability_chart_xgb.png")
    
    # Close plot to free up memory
    plt.close(fig)

def run_historical_test():
    try:
        # NOTE: Load the model you want to test.
        # Use 'nba_win_predictor.joblib' for Logistic Regression
        # Use 'nba_win_predictor_rf.joblib' for Random Forest
        model = joblib.load('nba_win_predictor_xgb.joblib')
        print("Successfully loaded 'nba_win_predictor_xgb.joblib'")
    except FileNotFoundError as e:
        print(f"Error: Could not load model file. {e}")
        return

    finished_games = find_recent_finished_games()
    if finished_games.empty:
        finished_games = find_games_on_sample_date()
    if finished_games.empty:
        print("\nCould not find any completed games.")
        return

    print("\n--- Completed Games Found ---")
    finished_games['GAME_DESC'] = finished_games.apply(lambda row: f"{row['AWAY_TEAM_ABBREVIATION']} @ {row['HOME_TEAM_ABBREVIATION']} ({row['GAME_STATUS_TEXT']})", axis=1)
    for idx, desc in enumerate(finished_games['GAME_DESC']):
         print(f"{idx}: {desc}")

    try:
        choice_idx = int(input("Select a game index to simulate: "))
        selected_game = finished_games.iloc[choice_idx]
    except (ValueError, IndexError):
        print("Invalid choice. Please exit and try again.")
        return

    game_id = selected_game['GAME_ID']
    home_team_name = selected_game['HOME_TEAM_ABBREVIATION']
    away_team_name = selected_game['AWAY_TEAM_ABBREVIATION']
    
    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id, timeout=60)
        pbp_df = pbp.get_data_frames()[0]
        pbp_df.reset_index(inplace=True)
        pbp_df.rename(columns={'index': 'PLAY_NUMBER'}, inplace=True)
        print(f"\nFound {len(pbp_df)} plays for {away_team_name} @ {home_team_name}. Starting simulation...")
        time.sleep(2)
    except Exception as e:
        print(f"Failed to fetch play-by-play data for game {game_id}: {e}")
        return

    home_score, away_score = 0, 0
    home_win_probabilities = []
    home_scores = []
    away_scores = []

    for index, play in pbp_df.iterrows():
        if play['SCORE'] and isinstance(play['SCORE'], str):
            away_score_str, home_score_str = play['SCORE'].split(' - ')
            home_score, away_score = int(home_score_str), int(away_score_str)

        home_scores.append(home_score)
        away_scores.append(away_score)

        period = play['PERIOD']
        time_str = play['PCTIMESTRING']
        score_margin = home_score - away_score
        seconds_remaining = parse_time_remaining(time_str, period)
        
        score_margin_sq = score_margin ** 2
        seconds_remaining_sq = seconds_remaining ** 2

        live_data = pd.DataFrame(
            [[score_margin, seconds_remaining, period, score_margin_sq, seconds_remaining_sq]],
            columns=['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']
        )
        
        win_proba = model.predict_proba(live_data)[0][1]
        home_win_probabilities.append(win_proba)

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- Simulating Game: {away_team_name} @ {home_team_name} ---")
        print(f"Play #{play['PLAY_NUMBER']}/{len(pbp_df)}")
        print(f"\nScore: {away_team_name} {away_score} - {home_score} {home_team_name}")
        print(f"Time:  {time_str} - Period {period}")
        event_desc = play['HOMEDESCRIPTION'] if play['HOMEDESCRIPTION'] else play['VISITORDESCRIPTION']
        print(f"Event: {event_desc}")
        print("\n--- Win Probability ---")
        print(f"  {home_team_name} (Home): {win_proba:.2%}")
        print(f"  {away_team_name} (Away): {1-win_proba:.2%}")
        time.sleep(0.05)

    print("\n--- Simulation Complete ---")
    if pd.notna(pbp_df['SCORE'].iloc[-1]):
        final_score_str = pbp_df['SCORE'].iloc[-1]
        away_final, home_final = map(int, final_score_str.split(' - '))
        print(f"Final Score: {away_team_name} {away_final} - {home_final} {home_team_name}")
        if home_win_probabilities:
            if home_final > away_final:
                home_win_probabilities[-1] = 1.0
            else:
                home_win_probabilities[-1] = 0.0
    else:
        print("Could not determine final score from data.")

    # Call the updated plotting function
    plot_win_probability(home_win_probabilities, home_scores, away_scores, home_team_name, away_team_name)

if __name__ == '__main__':
    run_historical_test()
