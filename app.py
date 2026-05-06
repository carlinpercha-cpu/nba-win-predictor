from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

def load_model(sport):
    model    = joblib.load(os.path.join(MODEL_DIR, f'{sport}_model.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, f'{sport}_scaler.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, f'{sport}_features.pkl'))
    return model, scaler, features

# Load all models at startup
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
        'model_performance': {
            'nba':          {'auc': 0.727, 'accuracy': 0.665},
            'nfl':          {'auc': 0.710, 'accuracy': 0.660},
            'mlb':          {'auc': 0.617, 'accuracy': 0.592},
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