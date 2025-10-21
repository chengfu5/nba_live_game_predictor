import dash
from dash import dcc, html, clientside_callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import requests
import re

# --- 1. Initialize Dash App and Load Model ---
app = dash.Dash(__name__)
server = app.server

try:
    model = joblib.load('nba_win_predictor_xgb.joblib')
    print("Successfully loaded 'nba_win_predictor_xgb.joblib'")
except FileNotFoundError:
    print("FATAL ERROR: Model file not found. Dashboard cannot make predictions.")
    model = None

# --- 2. Helper Functions ---
def parse_time_remaining(time_str, period):
    if pd.isna(time_str) or not isinstance(time_str, str): return 0
    if time_str.startswith('PT'):
        try:
            mins_match, secs_match = re.search(r'(\d+)M', time_str), re.search(r'(\d+)\.?\d*S', time_str)
            mins, secs = int(mins_match.group(1)) if mins_match else 0, int(secs_match.group(1)) if secs_match else 0
        except (AttributeError, ValueError): return 0
    elif ':' in time_str:
        try: mins, secs = map(int, time_str.split(':'))
        except ValueError: return 0
    else: return 0
    seconds_in_period = mins * 60 + secs
    return seconds_in_period + (4 - period) * 720 if period <= 4 else seconds_in_period

def find_games_on_date(date_str, find_finished_games=False):
    """Fetches game data from the NBA API and returns a list of game dicts."""
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
        data_frames = board.get_data_frames()
        games_df = data_frames[0]
        
        if games_df.empty:
            return []

        final_df = games_df.copy()

        # Defensively check for and merge standings data
        standings_available = False
        if len(data_frames) > 5:
            east_standings = data_frames[4]
            west_standings = data_frames[5]
            # Check if the expected columns exist before proceeding
            if 'WINS' in east_standings.columns and 'WINS' in west_standings.columns:
                standings_available = True
        
        if standings_available:
            all_standings = pd.concat([east_standings[['TEAM_ID', 'WINS', 'LOSSES']], west_standings[['TEAM_ID', 'WINS', 'LOSSES']]])
            # Merge records for the home team
            final_df = pd.merge(final_df, all_standings, left_on='HOME_TEAM_ID', right_on='TEAM_ID', how='left')
            final_df.rename(columns={'WINS': 'HOME_WINS', 'LOSSES': 'HOME_LOSSES'}, inplace=True)
            # Merge records for the visitor team
            final_df = pd.merge(final_df, all_standings, left_on='VISITOR_TEAM_ID', right_on='TEAM_ID', how='left')
            final_df.rename(columns={'WINS': 'AWAY_WINS', 'LOSSES': 'AWAY_LOSSES'}, inplace=True)

        # Fill missing records with 0 (for start of season) and convert to int
        for col in ['HOME_WINS', 'HOME_LOSSES', 'AWAY_WINS', 'AWAY_LOSSES']:
            if col not in final_df.columns:
                final_df[col] = 0
            final_df[col] = final_df[col].fillna(0).astype(int)

        # The abbreviations are often directly available in the main games_df
        # This is more reliable than merging, especially for scheduled games.
        # Get abbreviations as a fallback
        if 'HOME_TEAM_ABBREVIATION' not in final_df.columns:
            final_df[['VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_ABBREVIATION']] = final_df['GAMECODE'].str.split('/', expand=True)[1].str.findall(r'[A-Z]{3}').tolist()

        if find_finished_games:
            target_games = final_df[final_df['GAME_STATUS_TEXT'].str.contains('Final')]
        else:
            target_games = final_df[~final_df['GAME_STATUS_TEXT'].str.contains('Final')]
            
        return target_games.to_dict('records')
    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
        return []

def get_team_logo_url(team_tricode):
    """Constructs a more reliable URL for a team's logo."""
    # This URL format from stats.nba.com is generally more stable.
    if team_tricode in ['HOME', 'AWAY', 'N/A']: # Handle placeholder tricodes
        return "" # Return an empty string to avoid a broken image
    return f"https://stats.nba.com/media/img/teams/logos/{team_tricode}_logo.svg"

# --- NEW: Function to format the clock string for display ---
def format_clock_string(time_str):
    """Converts a time string ('PT05M32.0S' or '5:32') to a simple MM:SS format."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return "00:00"
    
    if time_str.startswith('PT'):
        try:
            mins_match = re.search(r'(\d+)M', time_str)
            secs_match = re.search(r'(\d+)\.?\d*S', time_str)
            mins = int(mins_match.group(1)) if mins_match else 0
            secs = int(secs_match.group(1)) if secs_match else 0
            # Format seconds with a leading zero if needed
            return f"{mins}:{secs:02d}"
        except (AttributeError, ValueError):
            return "00:00"
    elif ':' in time_str:
        return time_str 
    else:
        return "00:00"


# --- MODIFIED: The find_games_for_dropdown function now handles the "NBA Day" correctly ---
def find_games_for_dropdown():
    """Finds games for the current "NBA Day" or falls back to a historical date."""
    # The "NBA Day" runs until the early morning hours (e.g., 6 AM ET).
    # If the current time is before 6 AM, we should still look at yesterday's schedule
    # to catch late West Coast games.
    now = datetime.now()
    if now.hour < 6:
        target_date = now - timedelta(days=1)
    else:
        target_date = now
    
    target_date_str = target_date.strftime('%Y-%m-%d')
    print(f"Fetching games for NBA day: {target_date_str}")

    games = find_games_on_date(target_date_str)
    
    if not games:
        print(f"No games found for {target_date_str}, using fallback date.")
        fallback_date = '2024-04-10'
        games = find_games_on_date(fallback_date, find_finished_games=True)
        if not games:
            return [{'label': 'No games found', 'value': 'NONE', 'home_tricode': 'N/A', 'away_tricode': 'N/A', 'status': '', 'home_wins': 0, 'home_losses': 0, 'away_wins': 0, 'away_losses': 0}]
    
    return [{
        'label': f"{game['VISITOR_TEAM_ABBREVIATION']} @ {game['HOME_TEAM_ABBREVIATION']} ({game['GAME_STATUS_TEXT']})",
        'value': game['GAME_ID'],
        'home_tricode': game['HOME_TEAM_ABBREVIATION'],
        'away_tricode': game['VISITOR_TEAM_ABBREVIATION'],
        'status': game['GAME_STATUS_TEXT'],
        'home_wins': game.get('HOME_WINS', 0),
        'home_losses': game.get('HOME_LOSSES', 0),
        'away_wins': game.get('AWAY_WINS', 0),
        'away_losses': game.get('AWAY_LOSSES', 0),
    } for game in games]

# --- 3. Initial Data Fetch for Layout ---
GAMES_TODAY = find_games_for_dropdown() # This now uses the new logic
if not GAMES_TODAY:
    print("No games today, using fallback date for initial layout.")
    GAMES_TODAY = find_games_on_date('2024-04-10', find_finished_games=True)

GAME_INFO_MAP = {game['value']: { # Note: 'value' is the game_id from the dropdown options
    'home': game['label'].split(' @ ')[1].split(' (')[0],
    'away': game['label'].split(' @ ')[0]
} for game in GAMES_TODAY if game['value'] != 'NONE'}

# --- MODIFIED: Construct the tab labels with team records ---
game_tabs = []
for game in GAMES_TODAY:
    if game['value'] == 'NONE':
        game_tabs.append(dcc.Tab(label="No Games Found", value="NONE"))
        continue
    
    # Conditionally create the record string if the team has played
    away_record = ""
    if game['away_wins'] > 0 or game['away_losses'] > 0:
        away_record = f" ({game['away_wins']}-{game['away_losses']})"
    else:
        away_record = " (0-0)"

    home_record = ""
    if game['home_wins'] > 0 or game['home_losses'] > 0:
        home_record = f" ({game['home_wins']}-{game['home_losses']})"
    else:
        home_record = " (0-0)"

    label_text = f"{game['away_tricode']}{away_record} @ {game['home_tricode']}{home_record}"
    
    status_text = game['status']
    if ':' in status_text:
        label_text += f" ({status_text})"
        
    game_tabs.append(dcc.Tab(label=label_text, 
                             value=game['value'], 
                             style={'padding': '1rem', 'fontWeight': '500'}, 
                             selected_style={'padding': '1rem', 'fontWeight': 'bold', 'borderBottom': '3px solid #3B82F6'}))


# --- 4. Define the App Layout ---
todays_date_str = datetime.now().strftime('%B %d, %Y')
app.layout = html.Div(style={'fontFamily': 'Inter, sans-serif', 'backgroundColor': "#a4c1e8", 'padding': '2rem', 'minHeight': '100vh'}, children=[
    html.H1("Live NBA Scoreboard", style={'textAlign': 'center', 'color': '#111827'}),
    html.P(id='subtitle-date', style={'textAlign': 'center', 'color': '#4B5563', 'marginBottom': '2rem'}),
    dcc.Tabs(
        id="game-tabs",
        value=GAMES_TODAY[0]['value'] if GAMES_TODAY else None,
        children=game_tabs,
        style={'maxWidth': '1200px', 'margin': '0 auto'},
        content_style={'display': 'flex', 'flexWrap': 'nowrap', 'overflowX': 'auto'}
    ),
    
    # --- MODIFIED: Wrap plots in a styled card ---
    html.Div(style={'backgroundColor': 'white', 'borderRadius': '1.5rem', 'padding': '2rem', 'maxWidth': '1200px', 'margin': '2rem auto', 'boxShadow': '0 10px 15px -3px rgba(0, 0, 0, 0.1)'}, children=[
        html.Div(id='win-prob-display', style={'textAlign': 'center', 'fontSize': '2.25rem', 'fontWeight': 'bold', 'color': '#1E40AF'}),
        dcc.Graph(id='win-prob-chart'),
        html.Div(id='score-display', style={'textAlign': 'center', 'fontSize': '2.25rem', 'fontWeight': 'bold', 'marginTop': '2rem', 'color': '#1F2937'}),
        # --- NEW: Div to display the live commentary ---
        html.Div(id='commentary-display', style={'textAlign': 'center', 'fontSize': '1.1rem', 'color': '#4B5563', 'marginTop': '0.5rem', 'minHeight': '2rem'}),

        dcc.Graph(id='score-trend-chart'),
    ]),
    
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0)
])

# --- 5. Define the Callback for Live Updates ---
clientside_callback(
    """
    function(n) {
        const today = new Date();
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        const dateString = today.toLocaleDateString('en-US', options);
        return `Today's games: ${dateString}`;
    }
    """,
    Output('subtitle-date', 'children'),
    Input('game-tabs', 'value') 
)

@app.callback(
    [Output('win-prob-chart', 'figure'),
     Output('score-trend-chart', 'figure'),
     Output('win-prob-display', 'children'),
     Output('score-display', 'children'),
     Output('commentary-display', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('game-tabs', 'value')]
)
def update_live_charts(n, game_id):
    if not game_id:
        empty_fig = go.Figure().update_layout(title="Please select a game to track", xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return empty_fig, empty_fig, "", ""

    try:
        live_url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
        response = requests.get(live_url, timeout=10).json()
        df = pd.DataFrame(response.get('game', {}).get('actions', []))

        if df.empty:
            empty_fig = go.Figure().update_layout(title="Waiting for game to start or for first play to be logged...", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return empty_fig, empty_fig, "Awaiting Data", "Awaiting Data"

        df['HOME_SCORE'] = pd.to_numeric(df['scoreHome']).ffill().fillna(0).astype(int)
        df['AWAY_SCORE'] = pd.to_numeric(df['scoreAway']).ffill().fillna(0).astype(int)
        df.rename(columns={'period': 'PERIOD'}, inplace=True)
        df['SCORE_MARGIN'] = df['HOME_SCORE'] - df['AWAY_SCORE']
        df['SECONDS_REMAINING'] = df.apply(lambda row: parse_time_remaining(row['clock'], row['PERIOD']), axis=1)
        df['SCORE_MARGIN_SQ'] = df['SCORE_MARGIN'] ** 2
        df['SECONDS_REMAINING_SQ'] = df['SECONDS_REMAINING'] ** 2
        features = df[['SCORE_MARGIN', 'SECONDS_REMAINING', 'PERIOD', 'SCORE_MARGIN_SQ', 'SECONDS_REMAINING_SQ']]
        prob_history = model.predict_proba(features)[:, 1].tolist() if model else [0.5] * len(df)
        
        home_team = GAME_INFO_MAP.get(game_id, {}).get('home', 'Home')
        away_team = GAME_INFO_MAP.get(game_id, {}).get('away', 'Away')
        
        last_play = df.iloc[-1]
        home_score, away_score = last_play['HOME_SCORE'], last_play['AWAY_SCORE']
        home_win_prob = prob_history[-1]
        
        # --- NEW: Get commentary from the 'description' field ---
        commentary_text = last_play.get('description', '')
        is_game_over = last_play.get('actionType') == 'game' and last_play.get('subType') == 'end'
        if is_game_over:
            home_win_prob = 1.0 if home_score > away_score else 0.0
            prob_history[-1] = home_win_prob
            commentary_text = "Final"

        favored_team, favored_prob = (home_team, home_win_prob) if home_win_prob >= 0.5 else (away_team, 1 - home_win_prob)
        win_prob_text = f"{favored_team} Win Probability: {favored_prob:.1%}"
        
        # --- MODIFIED: Use the new helper function to format the clock string ---
        simple_clock = format_clock_string(last_play['clock'])
        time_display = f" {simple_clock} - Q{last_play['PERIOD']}"
        score_text = f"FINAL: {away_team} {away_score} @ {home_score} {home_team}" if is_game_over else f"{away_team} {away_score} @ {home_score} {home_team} | Time: {time_display}"

        period_end_indices = df.groupby('PERIOD').tail(1).index.tolist()
        period_end_plays = [i + 1 for i in period_end_indices]
        period_numbers = df.loc[period_end_indices, 'PERIOD'].tolist()
        tick_labels = [f"Q{p}" if p <= 4 else f"OT{p-4}" for p in period_numbers]
        shapes = [dict(type='line', xref='x', yref='paper', x0=p, y0=0, x1=p, y1=1, line=dict(color='lightgrey', width=1, dash='dash')) for p in period_end_plays[:-1]]

        play_numbers = list(range(1, len(prob_history) + 1))
        
        # --- MODIFIED: Updated plot styling ---
        plot_font_color = '#1F2937'

        win_prob_fig = go.Figure()
        win_prob_fig.add_trace(go.Scatter(x=play_numbers, y=prob_history, name=f'{home_team} Win Prob', line=dict(color='royalblue', width=3)))
        win_prob_fig.update_layout(
            title_text=f"<b>Win Probability Trend</b>", xaxis_title="Game Progression", yaxis_title="<b>Win Probability</b>", 
            yaxis_range=[0, 1], yaxis_tickformat=".0%", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=shapes, xaxis=dict(tickmode='array', tickvals=period_end_plays, ticktext=tick_labels),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=plot_font_color
        )
        
        score_trend_fig = go.Figure()
        score_trend_fig.add_trace(go.Scatter(x=play_numbers, y=df['HOME_SCORE'], name=f'{home_team} Score', line=dict(color='green')))
        score_trend_fig.add_trace(go.Scatter(x=play_numbers, y=df['AWAY_SCORE'], name=f'{away_team} Score', line=dict(color='red')))
        score_trend_fig.update_layout(
            title_text=f"<b>Score Progression</b>", xaxis_title="Game Progression", yaxis_title="<b>Score</b>", 
            yaxis_rangemode='tozero', 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=shapes, xaxis=dict(tickmode='array', tickvals=period_end_plays, ticktext=tick_labels),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=plot_font_color
        )

        return win_prob_fig, score_trend_fig, win_prob_text, score_text, commentary_text
    except Exception as e:
        error_fig = go.Figure().update_layout(title="Game not started")
        return error_fig, error_fig, "Game not started", "Game not started", "Game not started"

# --- 6. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


