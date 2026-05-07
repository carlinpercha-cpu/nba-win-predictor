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
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GOOGLE_SHEETS_ID = os.environ.get('GOOGLE_SHEETS_ID', '')
GOOGLE_SERVICE_ACCOUNT = os.environ.get('GOOGLE_SERVICE_ACCOUNT', '')

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

def log_prediction(sport, home_team, away_team, home_prob, away_prob, game_id):
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return
    try:
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            sport.upper(),
            home_team,
            away_team,
            round(home_prob * 100, 1),
            round(away_prob * 100, 1),
            'home' if home_prob > 0.5 else 'away',
            game_id,
            'pending'  # result updated later
        ]
        sheets_service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEETS_ID,
            range='Sheet1!A:I',
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

print("Loading models...")
models = {}
for sport in ['nba', 'nfl', 'mlb', 'ncaab', 'ncaab_bracket', 'cfb', 'nhl']:
    try:
        models[sport] = load_model(sport)
        print(f"  {sport}: loaded ({len(models[sport][2])} features)")
    except Exception as e:
        print(f"  {sport}: FAILED — {e}")
print("All models loaded.")

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'sheets_connected': sheets_service is not None,
        'model_performance': {
            'nba':          {'auc': 0.727, 'accuracy': 0.665},
            'nfl':          {'auc': 0.710, 'accuracy': 0.660},
            'mlb':          {'auc': 0.617, 'accuracy': 0.581},
            'ncaab':        {'auc': 0.862, 'accuracy': 0.774},
            'ncaab_bracket':{'auc': 0.927, 'accuracy': 0.839},
            'cfb':          {'auc': 0.867, 'accuracy': 0.776},
            'nhl':          {'auc': 0.749, 'accuracy': 0.675},
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sport = data.get('sport', 'nba').lower()

    if sport not in models:
        return jsonify({'error': f'Unknown sport: {sport}. Available: {list(models.keys())}'}), 400

    model, scaler, features = models[sport]

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
            log_prediction(sport, home_team, away_team, float(prob), 1-float(prob), game_id)

        return jsonify({
            'sport': sport,
            'win_probability': round(float(prob), 4),
            'win_percentage': round(float(prob) * 100, 1),
            'features_used': len(features),
            'missing_features': missing,
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
            range='Sheet1!A:I'
        ).execute()
        rows = result.get('values', [])
        predictions = []
        for row in rows[1:]:
            if len(row) >= 9:
                predictions.append({
                    'date': row[0],
                    'sport': row[1],
                    'home_team': row[2],
                    'away_team': row[3],
                    'home_prob': row[4],
                    'away_prob': row[5],
                    'prediction': row[6],
                    'game_id': row[7],
                    'result': row[8]
                })
        correct = sum(1 for p in predictions if p['result'] == p['prediction'])
        total_decided = sum(1 for p in predictions if p['result'] != 'pending')
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'decided': total_decided,
            'correct': correct,
            'accuracy': round(correct/total_decided*100, 1) if total_decided > 0 else None
        })
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
        sport_apis = {
            'NBA': ('basketball', 'nba'),
            'NFL': ('football', 'nfl'),
            'MLB': ('baseball', 'mlb'),
            'NHL': ('hockey', 'nhl'),
            'NCAAB': ('basketball', 'mens-college-basketball'),
            'CFB': ('football', 'college-football'),
        }
        for i, row in enumerate(rows[1:], start=2):
            if len(row) < 9 or row[8] != 'pending':
                continue
            sport, home_team, away_team, prediction, game_id = row[1], row[2], row[3], row[6], row[7]
            if sport not in sport_apis:
                continue
            sport_path, league = sport_apis[sport]
            try:
                espn_id = game_id.split('_')[1] if '_' in game_id else game_id
                r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/{sport_path}/{league}/summary?event={espn_id}', timeout=10)
                d = r.json()
                comp = d.get('header', {}).get('competitions', [{}])[0]
                status = comp.get('status', {}).get('type', {}).get('completed', False)
                if not status:
                    continue
                competitors = comp.get('competitors', [])
                home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                if not home or not away:
                    continue
                home_score = int(home.get('score', 0))
                away_score = int(away.get('score', 0))
                actual_winner = 'home' if home_score > away_score else 'away'
                sheets_service.spreadsheets().values().update(
                    spreadsheetId=GOOGLE_SHEETS_ID,
                    range=f'Sheet1!I{i}',
                    valueInputOption='RAW',
                    body={'values': [[actual_winner]]}
                ).execute()
                updated += 1
            except Exception as e:
                print(f"Update error for {game_id}: {e}")
        return jsonify({'updated': updated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bets', methods=['POST'])
def add_bet():
    if not sheets_service or not GOOGLE_SHEETS_ID:
        return jsonify({'error': 'Sheets not configured'}), 500
    try:
        data = request.get_json()
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
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
            {'key': 'nfl',           'name': 'NFL Football',           'auc': 0.710},
            {'key': 'mlb',           'name': 'MLB Baseball',           'auc': 0.617},
            {'key': 'ncaab',         'name': 'College Basketball',     'auc': 0.862},
            {'key': 'ncaab_bracket', 'name': 'March Madness Bracket',  'auc': 0.927},
            {'key': 'cfb',           'name': 'College Football',       'auc': 0.867},
            {'key': 'nhl',           'name': 'NHL Hockey',             'auc': 0.749},
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)