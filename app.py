from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
import json
from datetime import datetime
import google.auth
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=['GET', 'POST', 'PATCH', 'OPTIONS'], allow_headers=['Content-Type'])

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GOOGLE_SHEETS_ID = os.environ.get('GOOGLE_SHEETS_ID', '')
GOOGLE_SERVICE_ACCOUNT = os.environ.get('GOOGLE_SERVICE_ACCOUNT', '')
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '465c281c797b25ab75b326e3e0828a9f')
BDL_API_KEY = os.environ.get('BDL_API_KEY', '0a853bd2-7b7d-48f8-9bb4-aa7f6e74a613')

# Google Sheets setup
sheets_service = None
try:
    if GOOGLE_SERVICE_ACCOUNT:
        creds_dict = json.loads(GOOGLE_SERVICE_ACCOUNT)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        sheets_service = build('sheets', 'v4', credentials=creds)
        print("Google Sheets connected")
except Exception as e:
    print(f"Google Sheets failed: {e}")

def log_prediction(sport, home_team, away_team, home_prob, away_prob, game_id, vegas_prob=None):
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return
    try:
        # Check if this game_id is already logged
        existing = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!H:H'
        ).execute().get('values', [])
        existing_ids = {row[0] for row in existing[1:] if row}
        if game_id in existing_ids:
            return  # already logged, skip
        
        # Use US Eastern time so predictions are logged with the correct sport date
        from datetime import timezone, timedelta
        et_now = datetime.now(timezone.utc) - timedelta(hours=4)  # EDT (use 5 in winter for EST)
        row = [
            et_now.strftime('%Y-%m-%d'),
            sport.upper(),
            home_team,
            away_team,
            round(home_prob * 100, 1),
            round(away_prob * 100, 1),
            'home' if home_prob > 0.5 else 'away',
            game_id,
            'pending',
            round(vegas_prob * 100, 1) if vegas_prob else '',
            ''  # closing line filled in later
        ]
        sheets_service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:K',
            valueInputOption='RAW',
            body={'values': [row]}
        ).execute()
    except Exception as e:
        print(f"Sheets log error: {e}")

def load_model(sport):
    model    = joblib.load(os.path.join(MODEL_DIR, f'{sport}_model.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, f'{sport}_scaler.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, f'{sport}_features.pkl'))
    return model, scaler, features

models = {}

def get_model(sport):
    if sport not in models:
        try:
            models[sport] = load_model(sport)
        except Exception as e:
            return None
    return models[sport]

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'sheets_connected': sheets_service is not None,
        'model_performance': {
            'nba':          {'auc': 0.727, 'accuracy': 0.665},
            'nfl':          {'auc': 0.788, 'accuracy': 0.660},
            'mlb':          {'auc': 0.617, 'accuracy': 0.581},
            'ncaab':        {'auc': 0.862, 'accuracy': 0.774},
            'ncaab_bracket':{'auc': 0.927, 'accuracy': 0.839},
            'cfb':          {'auc': 0.867, 'accuracy': 0.776},
            'nhl':          {'auc': 0.749, 'accuracy': 0.675},
            'epl':          {'auc': 0.707, 'accuracy': 0.579},
            'tennis':       {'auc': 0.695, 'accuracy': 0.640},
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sport = data.get('sport', 'nba').lower()

    loaded = get_model(sport)
    if loaded is None:
        return jsonify({'error': f'Model unavailable: {sport}'}), 400
    model, scaler, features = loaded
    try:
        feature_vector = []
        missing = []
        for f in features:
            val = data.get(f, 0)
            if val is None:
                val = 0
                missing.append(f)
            feature_vector.append(float(val))

        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        # Log prediction if game_id provided
        game_id = data.get('game_id', '')
        home_team = data.get('home_team', '')
        away_team = data.get('away_team', '')
        if game_id and home_team and away_team:
            vegas_prob = data.get('vegas_prob') or data.get('Vegas_WinProb') or data.get('vegas_home_prob') or data.get('elo_win_prob') or data.get('BARTHAG')
            log_prediction(sport, home_team, away_team, float(prob), 1-float(prob), game_id, vegas_prob=float(vegas_prob) if vegas_prob else None)

        return jsonify({
            'sport': sport,
            'win_probability': round(float(prob), 4),
            'win_percentage': round(float(prob) * 100, 1),
            'features_used': len(features),
            'missing_features': missing,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_3way', methods=['POST'])
def predict_3way():
    data = request.get_json()
    sport = data.get('sport', 'epl').lower()
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f'{sport}_3way_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, f'{sport}_3way_scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_DIR, f'{sport}_3way_features.pkl'))
        
        feature_vector = [float(data.get(f, 0)) for f in features]
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]
        
        return jsonify({
            'sport': sport,
            'home_win_prob': round(float(probs[0]), 4),
            'draw_prob': round(float(probs[1]), 4),
            'away_win_prob': round(float(probs[2]), 4),
            'home_win_pct': round(float(probs[0]) * 100, 1),
            'draw_pct': round(float(probs[1]) * 100, 1),
            'away_win_pct': round(float(probs[2]) * 100, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/claude', methods=['POST'])
def claude_proxy():
    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'API key not configured'}), 500
    try:
        body = request.get_json()
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
            },
            json=body,
            timeout=60
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/record', methods=['GET'])
def get_record():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:K'
        ).execute()
        rows = result.get('values', [])
        predictions = []
        clv_values = []
        for row in rows[1:]:
            if len(row) >= 9:
                vegas_at_pred = row[9] if len(row) >= 10 and row[9] not in ('', None) else None
                closing_line = row[10] if len(row) >= 11 and row[10] not in ('', None) else None
                
                # Calculate CLV: positive if model picked side that line moved toward
                clv = None
                if vegas_at_pred and closing_line:
                    try:
                        v_pred = float(vegas_at_pred)
                        v_close = float(closing_line)
                        model_pick_home = float(row[4]) > 50
                        # If model picked home and home % rose at close = positive CLV
                        if model_pick_home:
                            clv = round(v_close - v_pred, 1)
                        else:
                            clv = round(v_pred - v_close, 1)
                        clv_values.append(clv)
                    except (ValueError, TypeError):
                        pass
                
                predictions.append({
                    'date': row[0],
                    'sport': row[1],
                    'home_team': row[2],
                    'away_team': row[3],
                    'home_prob': row[4],
                    'away_prob': row[5],
                    'prediction': row[6],
                    'game_id': row[7],
                    'result': row[8],
                    'vegas_at_pred': vegas_at_pred,
                    'closing_line': closing_line,
                    'clv': clv
                })
        
        correct = sum(1 for p in predictions if p['result'] == p['prediction'])
        total_decided = sum(1 for p in predictions if p['result'] != 'pending')
        avg_clv = round(sum(clv_values) / len(clv_values), 2) if clv_values else None
        positive_clv = sum(1 for c in clv_values if c > 0)
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'decided': total_decided,
            'correct': correct,
            'accuracy': round(correct/total_decided*100, 1) if total_decided > 0 else None,
            'avg_clv': avg_clv,
            'clv_count': len(clv_values),
            'clv_positive_pct': round(positive_clv/len(clv_values)*100, 1) if clv_values else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calibration', methods=['GET'])
def get_calibration():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:I'
        ).execute()
        rows = result.get('values', [])
        
        # Bin predictions by confidence level
        bins = [(0, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 80), (80, 100)]
        bin_data = [{'low': lo, 'high': hi, 'predicted_avg': 0, 'actual_winrate': 0, 'count': 0, 'wins': 0} for lo, hi in bins]
        
        for row in rows[1:]:
            if len(row) < 9 or row[8] == 'pending':
                continue
            try:
                # Use the predicted side's confidence
                home_prob = float(row[4])
                away_prob = float(row[5])
                prediction = row[6]
                result_actual = row[8]
                # Confidence on the side we picked
                conf = home_prob if prediction == 'home' else away_prob
                won = 1 if prediction == result_actual else 0
                
                for i, (lo, hi) in enumerate(bins):
                    if lo <= conf < hi or (hi == 100 and conf == 100):
                        bin_data[i]['count'] += 1
                        bin_data[i]['wins'] += won
                        bin_data[i]['predicted_avg'] += conf
                        break
            except (ValueError, IndexError):
                continue
        
        # Calculate averages
        for b in bin_data:
            if b['count'] > 0:
                b['predicted_avg'] = round(b['predicted_avg'] / b['count'], 1)
                b['actual_winrate'] = round(b['wins'] / b['count'] * 100, 1)
            else:
                b['predicted_avg'] = (b['low'] + b['high']) / 2
                b['actual_winrate'] = None
        
        return jsonify({'calibration': bin_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_results', methods=['POST'])
def update_results():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:I'
        ).execute()
        rows = result.get('values', [])
        updated = 0
        skipped = 0
        max_updates = 20  # cap per run to stay under Render's 30s timeout
        sport_apis = {
            'NBA': ('basketball', 'nba'),
            'NFL': ('football', 'nfl'),
            'MLB': ('baseball', 'mlb'),
            'NHL': ('hockey', 'nhl'),
            'NCAAB': ('basketball', 'mens-college-basketball'),
            'CFB': ('football', 'college-football'),
            'EPL': ('soccer', 'eng.1'),
        }
        for i, row in enumerate(rows[1:], start=2):
            if updated >= max_updates:
                break
            if len(row) < 9 or row[8] != 'pending':
                continue
            sport, home_team, away_team, prediction, game_id = row[1], row[2], row[3], row[6], row[7]
            if sport not in sport_apis:
                continue
            sport_path, league = sport_apis[sport]
            try:
                bdl_id = game_id.split('_')[1] if '_' in game_id else game_id
                
                if sport == 'NBA':
                    # Use BallDontLie API for NBA
                    r = requests.get(
                        f'https://api.balldontlie.io/v1/games/{bdl_id}',
                        headers={'Authorization': BDL_API_KEY},
                        timeout=8
                    )
                    d = r.json()
                    g = d.get('data', {})
                    if g.get('status') != 'Final':
                        skipped += 1
                        continue
                    home_score = g.get('home_team_score', 0)
                    away_score = g.get('visitor_team_score', 0)
                    if home_score > away_score:
                        actual_winner = 'home'
                    elif away_score > home_score:
                        actual_winner = 'away'
                    else:
                        actual_winner = 'draw'
                else:
                    # ESPN for everything else
                    r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/{sport_path}/{league}/summary?event={bdl_id}', timeout=8)
                    d = r.json()
                    comp = d.get('header', {}).get('competitions', [{}])[0]
                    status = comp.get('status', {}).get('type', {}).get('completed', False)
                    if not status:
                        skipped += 1
                        continue
                    competitors = comp.get('competitors', [])
                    home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                    away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                    if not home or not away:
                        continue
                    home_score = int(home.get('score', 0))
                    away_score = int(away.get('score', 0))
                    if home_score > away_score:
                        actual_winner = 'home'
                    elif away_score > home_score:
                        actual_winner = 'away'
                    else:
                        actual_winner = 'draw'
                sheets_service.spreadsheets().values().update(
                    spreadsheetId=GOOGLE_SHEETS_ID,
                    range=f'Sheet1!I{i}',
                    valueInputOption='RAW',
                    body={'values': [[actual_winner]]}
                ).execute()
                updated += 1
            except Exception as e:
                print(f"Update error for {game_id}: {e}")
        return jsonify({'updated': updated, 'skipped': skipped})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_closing_lines', methods=['POST'])
def update_closing_lines():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    
    ODDS_KEY = ODDS_API_KEY
    sport_keys = {
        'NBA': 'basketball_nba',
        'NFL': 'americanfootball_nfl',
        'MLB': 'baseball_mlb',
        'NHL': 'icehockey_nhl',
        'EPL': 'soccer_epl',
    }
    
    try:
        # Read sheet
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:K'
        ).execute()
        rows = result.get('values', [])
        
        # Find pending rows with no closing line yet
        pending_by_sport = {}
        for i, row in enumerate(rows[1:], start=2):
            if len(row) < 9:
                continue
            sport = row[1]
            if sport not in sport_keys:
                continue
            # Skip if closing line already set
            if len(row) >= 11 and row[10] not in (None, '', 'None'):
                continue
            home_team = row[2]
            away_team = row[3]
            pending_by_sport.setdefault(sport, []).append({
                'row_num': i,
                'home_team': home_team,
                'away_team': away_team,
            })
        
        if not pending_by_sport:
            return jsonify({'updated': 0, 'message': 'no pending closing lines'})
        
        updated = 0
        for sport, entries in pending_by_sport.items():
            try:
                odds_url = f'https://api.the-odds-api.com/v4/sports/{sport_keys[sport]}/odds/?apiKey={ODDS_KEY}&regions=us&markets=h2h&oddsFormat=american'
                r = requests.get(odds_url, timeout=10)
                games = r.json()
                if not isinstance(games, list):
                    continue
                
                # Build lookup: home_team|away_team -> implied prob
                odds_lookup = {}
                for g in games:
                    h = g.get('home_team', '')
                    a = g.get('away_team', '')
                    bookmakers = g.get('bookmakers', [])
                    if not bookmakers:
                        continue
                    # Use first bookmaker (DraftKings/FanDuel typically)
                    market = bookmakers[0].get('markets', [{}])[0]
                    outcomes = market.get('outcomes', [])
                    home_odds = next((o['price'] for o in outcomes if o['name'] == h), None)
                    if home_odds is None:
                        continue
                    # Convert American odds to implied probability
                    if home_odds > 0:
                        implied = 100 / (home_odds + 100)
                    else:
                        implied = -home_odds / (-home_odds + 100)
                    odds_lookup[f'{h}|{a}'] = implied
                
                # Update matching rows
                for entry in entries:
                    key = f'{entry["home_team"]}|{entry["away_team"]}'
                    if key not in odds_lookup:
                        continue
                    closing_prob = round(odds_lookup[key] * 100, 1)
                    sheets_service.spreadsheets().values().update(
                        spreadsheetId=GOOGLE_SHEETS_ID,
                        range=f'Sheet1!K{entry["row_num"]}',
                        valueInputOption='RAW',
                        body={'values': [[closing_prob]]}
                    ).execute()
                    updated += 1
                    if updated >= 20:
                        break
            except Exception as e:
                print(f"{sport} closing line error: {e}")
            
            if updated >= 20:
                break
        
        return jsonify({'updated': updated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refresh_stats', methods=['POST'])
def refresh_stats():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        results = {}
        sport_param = request.args.get('sport', 'all').lower()
        
        sport_configs = {
            'nba': ('basketball', 'nba'),
            'nfl': ('football', 'nfl'),
            'mlb': ('baseball', 'mlb'),
            'nhl': ('hockey', 'nhl'),
            'epl': ('soccer', 'eng.1'),
        }
        
        if sport_param == 'all':
            sports_to_run = list(sport_configs.keys())
        elif sport_param in sport_configs:
            sports_to_run = [sport_param]
        else:
            return jsonify({'error': f'Unknown sport: {sport_param}'}), 400
        
        for sport in sports_to_run:
            espn_sport, espn_league = sport_configs[sport]
            result = fetch_espn_team_stats(espn_sport, espn_league)
            stats = result.get('stats', [])
            results[f'{sport}_count'] = len(stats)
            if stats:
                write_team_stats(sport.upper(), stats)

        return jsonify({'status': 'ok', 'updated': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/team_stats', methods=['GET'])
def get_team_stats():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='TeamStats!A:F'
        ).execute()
        rows = result.get('values', [])
        stats = {}
        for row in rows[1:]:
            if len(row) < 5:
                continue
            sport, team_id, team_name = row[0], row[1], row[2]
            stats[f'{sport}|{team_name}'] = {
                'last5_winrate': float(row[3]) if row[3] else 0.5,
                'last10_winrate': float(row[4]) if row[4] else 0.5,
                'games_played': int(row[5]) if len(row) > 5 and row[5] else 0,
            }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def fetch_espn_team_stats(sport, league):
    """Generic ESPN team stats fetcher for any sport/league."""
    try:
        teams_resp = requests.get(
            f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams',
            timeout=15
        )
        teams_data = teams_resp.json()
        teams = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        team_stats = []
        for team_wrapper in teams:
            team = team_wrapper.get('team', {})
            team_id = team.get('id')
            team_name = team.get('displayName')
            if not team_id:
                continue

            try:
                schedule_resp = requests.get(
                    f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule',
                    timeout=15
                )
                events = schedule_resp.json().get('events', [])
                completed = []
                for event in events:
                    competitions = event.get('competitions', [{}])
                    if not competitions:
                        continue
                    comp = competitions[0]
                    is_completed = comp.get('status', {}).get('type', {}).get('completed', False)
                    if not is_completed:
                        continue
                    competitors = comp.get('competitors', [])
                    team_comp = next((c for c in competitors if str(c.get('team', {}).get('id')) == str(team_id)), None)
                    opp_comp = next((c for c in competitors if str(c.get('team', {}).get('id')) != str(team_id)), None)
                    if not team_comp or not opp_comp:
                        continue
                    team_score_str = team_comp.get('score', '0')
                    opp_score_str = opp_comp.get('score', '0')
                    if isinstance(team_score_str, dict):
                        team_score_str = team_score_str.get('value', '0')
                    if isinstance(opp_score_str, dict):
                        opp_score_str = opp_score_str.get('value', '0')
                    team_score = int(team_score_str) if team_score_str else 0
                    opp_score = int(opp_score_str) if opp_score_str else 0
                    won = 1 if team_score > opp_score else 0
                    completed.append({'won': won})

                if len(completed) == 0:
                    continue
                completed = completed[-10:]
                last5 = completed[-5:]
                wr5 = sum(g['won'] for g in last5) / len(last5)
                wr10 = sum(g['won'] for g in completed) / len(completed)
                team_stats.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'last5_winrate': round(wr5, 3),
                    'last10_winrate': round(wr10, 3),
                    'games_played': len(completed),
                })
            except Exception:
                continue

        return {'stats': team_stats}
    except Exception as e:
        return {'stats': [], 'error': str(e)}

def fetch_nba_team_stats():
    """Fetch last 10 games stats for each NBA team from ESPN."""
    debug_info = {'teams_found': 0, 'completed_games_total': 0, 'errors': []}
    try:
        teams_resp = requests.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams',
            timeout=15
        )
        teams_data = teams_resp.json()
        teams = teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
        debug_info['teams_found'] = len(teams)

        team_stats = []
        for team_wrapper in teams:
            team = team_wrapper.get('team', {})
            team_id = team.get('id')
            team_name = team.get('displayName')
            if not team_id:
                continue

            try:
                schedule_resp = requests.get(
                    f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule',
                    timeout=15
                )
                schedule_data = schedule_resp.json()
                events = schedule_data.get('events', [])
                debug_info[f'team_{team_name}_events'] = len(events)
                
                completed = []
                for event in events:
                    competitions = event.get('competitions', [{}])
                    if not competitions:
                        continue
                    comp = competitions[0]
                    status_obj = comp.get('status', {}).get('type', {})
                    is_completed = status_obj.get('completed', False)
                    if not is_completed:
                        continue
                    
                    competitors = comp.get('competitors', [])
                    team_comp = next((c for c in competitors if str(c.get('team', {}).get('id')) == str(team_id)), None)
                    opp_comp = next((c for c in competitors if str(c.get('team', {}).get('id')) != str(team_id)), None)
                    if not team_comp or not opp_comp:
                        continue
                    
                    team_score_str = team_comp.get('score', '0')
                    opp_score_str = opp_comp.get('score', '0')
                    if isinstance(team_score_str, dict):
                        team_score_str = team_score_str.get('value', '0')
                    if isinstance(opp_score_str, dict):
                        opp_score_str = opp_score_str.get('value', '0')
                    
                    team_score = int(team_score_str) if team_score_str else 0
                    opp_score = int(opp_score_str) if opp_score_str else 0
                    won = 1 if team_score > opp_score else 0
                    completed.append({'won': won, 'pts_for': team_score, 'pts_against': opp_score})
                
                debug_info[f'team_{team_name}_completed'] = len(completed)
                debug_info['completed_games_total'] += len(completed)
                
                if len(completed) == 0:
                    continue

                completed = completed[-10:]
                last5 = completed[-5:]
                wr5 = sum(g['won'] for g in last5) / len(last5)
                wr10 = sum(g['won'] for g in completed) / len(completed)
                
                team_stats.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'last5_winrate': round(wr5, 3),
                    'last10_winrate': round(wr10, 3),
                    'games_played': len(completed),
                })
            except Exception as team_err:
                debug_info['errors'].append(f'{team_name}: {str(team_err)[:100]}')

        return {'stats': team_stats, 'debug': debug_info}
    except Exception as e:
        debug_info['errors'].append(f'top_level: {str(e)}')
        return {'stats': [], 'debug': debug_info}


def write_team_stats(sport, stats):
    """Write team stats to TeamStats tab in Google Sheet."""
    try:
        # Clear existing rows for this sport
        all_rows = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='TeamStats!A:F'
        ).execute().get('values', [])

        # Keep header + non-matching sport rows
        header = ['Sport', 'Team ID', 'Team Name', 'Last5 WinRate', 'Last10 WinRate', 'Games Played']
        kept = [r for r in all_rows[1:] if len(r) > 0 and r[0] != sport]

        new_rows = [[sport, s['team_id'], s['team_name'], s['last5_winrate'],
                     s['last10_winrate'], s['games_played']] for s in stats]

        sheets_service.spreadsheets().values().clear(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='TeamStats!A:F'
        ).execute()

        sheets_service.spreadsheets().values().update(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='TeamStats!A1',
            valueInputOption='RAW',
            body={'values': [header] + kept + new_rows}
        ).execute()
    except Exception as e:
        print(f"Sheets write error: {e}")

@app.route('/bets', methods=['POST'])
def add_bet():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        data = request.get_json()
        from datetime import timezone, timedelta
        et_now = datetime.now(timezone.utc) - timedelta(hours=4)
        row = [
            et_now.strftime('%Y-%m-%d'),
            data.get('sport', ''),
            data.get('matchup', ''),
            data.get('pick', ''),
            data.get('odds', ''),
            data.get('stake', ''),
            'pending',
            ''
        ]
        sheets_service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Bets!A:H',
            valueInputOption='RAW',
            body={'values': [row]}
        ).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bets', methods=['GET'])
def get_bets():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Bets!A:H'
        ).execute()
        rows = result.get('values', [])
        bets = []
        total_stake = 0
        total_profit = 0
        wins = 0
        decided = 0
        for row in rows[1:]:
            if len(row) < 7:
                continue
            stake = float(row[5]) if row[5] else 0
            profit = float(row[7]) if len(row) > 7 and row[7] else 0
            bets.append({
                'date': row[0], 'sport': row[1], 'matchup': row[2],
                'pick': row[3], 'odds': row[4], 'stake': stake,
                'result': row[6] if len(row) > 6 else 'pending',
                'profit': profit
            })
            if row[6] != 'pending' and len(row) > 6:
                total_stake += stake
                total_profit += profit
                decided += 1
                if row[6] == 'win': wins += 1
        roi = round(total_profit/total_stake*100, 1) if total_stake > 0 else 0
        win_rate = round(wins/decided*100, 1) if decided > 0 else 0
        return jsonify({
            'bets': list(reversed(bets)),
            'total_bets': len(bets),
            'decided': decided,
            'wins': wins,
            'win_rate': win_rate,
            'total_profit': round(total_profit, 2),
            'total_staked': round(total_stake, 2),
            'roi': roi
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bets/<int:row_num>', methods=['PATCH'])
def update_bet(row_num):
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        data = request.get_json()
        result = data.get('result', 'pending')
        odds = float(data.get('odds', 0))
        stake = float(data.get('stake', 0))
        if result == 'win':
            profit = stake * (odds/100) if odds > 0 else stake * (100/abs(odds))
        elif result == 'loss':
            profit = -stake
        else:
            profit = 0
        sheets_service.spreadsheets().values().update(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range=f'Bets!G{row_num}:H{row_num}',
            valueInputOption='RAW',
            body={'values': [[result, round(profit, 2)]]}
        ).execute()
        return jsonify({'success': True, 'profit': round(profit, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features/<sport>')
def get_features(sport):
    if sport not in models:
        return jsonify({'error': f'Unknown sport: {sport}'}), 400
    _, _, features = models[sport]
    return jsonify({'sport': sport, 'features': features})

@app.route('/sports')
def get_sports():
    return jsonify({
        'sports': [
            {'key': 'nba',           'name': 'NBA Basketball',         'auc': 0.727},
            {'key': 'nfl',           'name': 'NFL Football',           'auc': 0.788},
            {'key': 'mlb',           'name': 'MLB Baseball',           'auc': 0.617},
            {'key': 'ncaab',         'name': 'College Basketball',     'auc': 0.862},
            {'key': 'ncaab_bracket', 'name': 'March Madness Bracket',  'auc': 0.927},
            {'key': 'cfb',           'name': 'College Football',       'auc': 0.867},
            {'key': 'nhl',           'name': 'NHL Hockey',             'auc': 0.749},
            {'key': 'epl',           'name': 'EPL Soccer',             'auc': 0.707},
            {'key': 'tennis',        'name': 'ATP Tennis',             'auc': 0.695},
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)