import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
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
        games, line_score = data_frames[0], data_frames[1]
        
        if games.empty:
            return []

        team_abbrevs = line_score[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
        merged_df = pd.merge(games, team_abbrevs, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'])
        merged_df.rename(columns={'TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}, inplace=True)
        final_merged_df = pd.merge(merged_df, team_abbrevs, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'])
        final_merged_df.rename(columns={'TEAM_ABBREVIATION': 'VISITOR_TEAM_ABBREVIATION'}, inplace=True)


        if find_finished_games:
            return final_merged_df[final_merged_df['GAME_STATUS_TEXT'].str.contains('Final')].to_dict('records')
        else:
            return final_merged_df[~final_merged_df['GAME_STATUS_TEXT'].str.contains('Final')].to_dict('records')
    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
        return []

# --- 3. Initial Data Fetch for Layout ---
GAMES_TODAY = find_games_on_date(datetime.now().strftime('%Y-%m-%d'))
if not GAMES_TODAY:
    print("No games today, using fallback date for initial layout.")
    GAMES_TODAY = find_games_on_date('2024-04-10', find_finished_games=True)

GAME_INFO_MAP = {game['GAME_ID']: {
    'home': game['HOME_TEAM_ABBREVIATION'],
    'away': game['VISITOR_TEAM_ABBREVIATION']
} for game in GAMES_TODAY}

game_tabs = []
for game in GAMES_TODAY:
    status_text = game['GAME_STATUS_TEXT']
    label_text = f"{game['VISITOR_TEAM_ABBREVIATION']} @ {game['HOME_TEAM_ABBREVIATION']}"
    if 'PM' in status_text or 'AM' in status_text:
        label_text += f" ({status_text})"
    game_tabs.append(dcc.Tab(
        label=label_text, value=game['GAME_ID'],
        style={'padding': '1rem', 'fontWeight': '500'},
        selected_style={'padding': '1rem', 'fontWeight': 'bold', 'borderBottom': '3px solid #3B82F6'}
    ))


# --- 4. Define the App Layout ---
app.layout = html.Div(style={'fontFamily': 'Inter, sans-serif', 'backgroundColor': "#a4c1e8", 'padding': '2rem', 'minHeight': '100vh'}, children=[
    html.H1("Live NBA Scoreboard", style={'textAlign': 'center', 'color': '#111827'}),
    html.P("Today's games", style={'textAlign': 'center', 'color': '#4B5563', 'marginBottom': '2rem'}),
    
    dcc.Tabs(id="game-tabs", value=GAMES_TODAY[0]['GAME_ID'] if GAMES_TODAY else None, children=game_tabs, style={'maxWidth': '1200px', 'margin': '0 auto'}),
    
    # --- MODIFIED: Wrap plots in a styled card ---
    html.Div(style={'backgroundColor': 'white', 'borderRadius': '1.5rem', 'padding': '2rem', 'maxWidth': '1200px', 'margin': '2rem auto', 'boxShadow': '0 10px 15px -3px rgba(0, 0, 0, 0.1)'}, children=[
        html.Div(id='win-prob-display', style={'textAlign': 'center', 'fontSize': '2.25rem', 'fontWeight': 'bold', 'color': '#1E40AF'}),
        dcc.Graph(id='win-prob-chart'),
        html.Div(id='score-display', style={'textAlign': 'center', 'fontSize': '2.25rem', 'fontWeight': 'bold', 'marginTop': '2rem', 'color': '#1F2937'}),
        dcc.Graph(id='score-trend-chart'),
    ]),
    
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0)
])

# --- 5. Define the Callback for Live Updates ---
@app.callback(
    [Output('win-prob-chart', 'figure'),
     Output('score-trend-chart', 'figure'),
     Output('win-prob-display', 'children'),
     Output('score-display', 'children')],
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
        
        is_game_over = last_play.get('actionType') == 'game' and last_play.get('subType') == 'end'
        if is_game_over:
            home_win_prob = 1.0 if home_score > away_score else 0.0
            prob_history[-1] = home_win_prob

        favored_team, favored_prob = (home_team, home_win_prob) if home_win_prob >= 0.5 else (away_team, 1 - home_win_prob)
        win_prob_text = f"{favored_team} Win Probability: {favored_prob:.1%}"
        score_text = f"FINAL: {away_team} {away_score} @ {home_score} {home_team}" if is_game_over else f"{away_team} {away_score} @ {home_score} {home_team}"

        period_end_indices = df.groupby('PERIOD').tail(1).index.tolist()
        period_end_plays = [i + 1 for i in period_end_indices]
        period_numbers = df.loc[period_end_indices, 'PERIOD'].tolist()
        tick_labels = [f"End Q{p}" if p <= 4 else f"End OT{p-4}" for p in period_numbers]
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

        return win_prob_fig, score_trend_fig, win_prob_text, score_text
    except Exception as e:
        error_fig = go.Figure().update_layout(title=f"An error occurred: {e}")
        return error_fig, error_fig, "Error", "Error"

# --- 6. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)




####### http://127.0.0.1:8050/ ####### This is the local address to access the Dash app.