import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('nba_sklearn_model.pkl')
scaler = joblib.load('nba_sklearn_scaler.pkl')
feature_cols = joblib.load('nba_feature_cols.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)
        return jsonify({
            'win_probability': round(float(proba[0][1]), 4),
            'lose_probability': round(float(proba[0][0]), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'NBA Win Predictor v2', 'auc': 0.7231})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
