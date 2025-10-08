import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2
import time
from requests.exceptions import ReadTimeout

def get_game_ids_for_season(season_str='2023-24'):
    """
    Retrieves all regular season game IDs for a given season.
    Example season_str: '2023-24'
    """
    print(f"Finding all game IDs for the {season_str} season...")
    game_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season_str,
        league_id_nullable='00', # NBA
        season_type_nullable='Regular Season'
    )
    games_dict = game_finder.get_normalized_dict()
    games = games_dict['LeagueGameFinderResults']
    game_ids = [game['GAME_ID'] for game in games]
    print(f"Found {len(game_ids)} games.")
    return list(set(game_ids)) # Use set to remove duplicates

def parse_time_remaining(time_str, period):
    """
    Converts clock time and period into total seconds remaining in the game.
    """
    if pd.isna(time_str):
        return 0
        
    mins, secs = map(int, time_str.split(':'))
    
    # Calculate seconds remaining in the current period
    seconds_in_period = mins * 60 + secs
    
    # Quarters are 12 minutes (720 seconds)
    # Overtime periods are 5 minutes (300 seconds)
    
    # If in regulation (periods 1-4)
    if period <= 4:
        full_periods_remaining = 4 - period
        return seconds_in_period + full_periods_remaining * 720
    # If in overtime
    else:
        # In OT, we assume the game could end at any moment, so we don't add future OT periods.
        # A more complex model might handle this differently.
        return seconds_in_period

def collect_season_data(game_ids):
    """
    Collects play-by-play data for a list of game IDs and processes it.
    """
    all_games_data = []
    # Using enumerate to get an index 'i' for each game
    for i, game_id in enumerate(game_ids):
        
        # --- NEW: Stop after 600 games ---
        # The index 'i' starts at 0, so when i is 600, it's the 601st game.
        if len(all_games_data) >= 600:
            print(f"\nSuccessfully collected 600 games. Stopping data collection.")
            break

        # Retry Logic
        for attempt in range(3):
            try:
                print(f"Processing game {len(all_games_data) + 1}/600: {game_id} (Attempt {attempt + 1})")
                
                # Increased Timeout
                pbp = playbyplayv2.PlayByPlayV2(game_id, timeout=60)
                pbp_df = pbp.get_data_frames()[0]

                if pbp_df.empty:
                    break 

                pbp_df[['AWAY_SCORE_STR', 'HOME_SCORE_STR']] = pbp_df['SCORE'].str.split(' - ', expand=True, n=1)
                pbp_df['AWAY_SCORE'] = pd.to_numeric(pbp_df['AWAY_SCORE_STR'], errors='coerce')
                pbp_df['HOME_SCORE'] = pd.to_numeric(pbp_df['HOME_SCORE_STR'], errors='coerce')
                
                pbp_df.fillna(method='ffill', inplace=True)

                pbp_df['SECONDS_REMAINING'] = pbp_df.apply(
                    lambda row: parse_time_remaining(row['PCTIMESTRING'], row['PERIOD']),
                    axis=1
                )
                
                final_home_score = pbp_df['HOME_SCORE'].iloc[-1]
                final_away_score = pbp_df['AWAY_SCORE'].iloc[-1]
                pbp_df['HOME_TEAM_WINS'] = 1 if final_home_score > final_away_score else 0

                game_data = pbp_df[[
                    'GAME_ID', 'PERIOD', 'SECONDS_REMAINING', 
                    'HOME_SCORE', 'AWAY_SCORE', 'HOME_TEAM_WINS'
                ]].copy()
                
                all_games_data.append(game_data)
                
                time.sleep(0.7)
                
                break

            except ReadTimeout:
                print(f"  Timeout occurred for game {game_id}. Waiting 10 seconds before retrying...")
                time.sleep(10)
                if attempt == 2:
                    print(f"  Could not retrieve data for game {game_id} after 3 attempts. Skipping.")

            except Exception as e:
                print(f"  An unexpected error occurred for game {game_id}: {e}. Skipping.")
                break

    if not all_games_data:
        print("No data was collected. Exiting.")
        return

    final_df = pd.concat(all_games_data, ignore_index=True)
    final_df.to_csv('data/nba_game_data.csv', index=False)
    print(f"\nData collection complete. Saved {len(all_games_data)} games to data/nba_game_data.csv")




if __name__ == '__main__':
    # You can change the season string to collect data from different seasons
    season_game_ids = get_game_ids_for_season('2023-24')
    # For a quicker test, you can slice the list: season_game_ids[:20]
    collect_season_data(season_game_ids)
