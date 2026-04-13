# 🏀 NBA Win Predictor AI

A full-stack AI system that predicts NBA game outcomes using machine learning trained on 30,000+ historical games. Built with Microsoft Azure ML, deployed as a live REST API, and served through a web application powered by Claude AI.

**Live Model API:** https://nba-win-predictor-airk.onrender.com/health

---

## Results

| Metric | V1 | V2 |
|--------|----|----|
| AUC | 0.661 | **0.723** |
| Accuracy | 61.6% | **67.8%** |
| F1 Score | 0.621 | **0.673** |
| Training Games | ~24,000 | **30,393** |

> Vegas oddsmakers with real-time injury data achieve ~67-70% accuracy. This model hits **67.8%** using only publicly available historical data.

---

## Project Structure

```
nba-win-predictor/
├── app.py                    # Flask API serving the model
├── requirements.txt          # Python dependencies
├── nba_ai_predictor_v4.html  # Web application (open in browser)
├── nba_sklearn_model.pkl     # Trained Logistic Regression model
├── nba_sklearn_scaler.pkl    # StandardScaler for feature normalization
├── nba_feature_cols.pkl      # Feature column list
└── README.md
```

---

## Architecture

```
Kaggle Data (3 datasets)
    ↓
Azure ML Notebook (Feature Engineering)
    ↓
Azure ML Designer (Train + Evaluate)
    ↓
scikit-learn export → Flask API → Render.com
    ↓
Web App (HTML) ← Claude API (analysis) + BallDontLie API (schedule)
```

---

## Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| NBA Team Traditional Stats | Kaggle | 1996–2024 |
| NBA Four Factors Dataset | Kaggle | 1996–2024 |
| NBA Betting Data | Kaggle | 2007–2025 |

---

## Features (39 Total)

All features are computed from historical pre-game data only — no same-game statistics are used to prevent data leakage.

**Four Factors (rolling 5 & 10-game averages)**
- `Last_5_EFG_PCT` — Effective Field Goal %
- `Last_5_FTA_RATE` — Free Throw Rate
- `Last_5_TOV_PCT` — Turnover %
- `Last_5_OREB_PCT` — Offensive Rebound %
- Opponent versions of all four factors

**Momentum & Form**
- `Last_5_WinRate`, `Last_10_WinRate` — Rolling win rates
- `Win_Streak` — Current win/loss streak (positive = winning)

**Situational**
- `is_home`, `is_away` — Home court indicator
- `Days_Rest` — Days since last game (capped at 14)
- `Is_B2B` — Back-to-back game flag
- `Rest_Differential` — Rest advantage vs opponent

**Matchup Differentials**
- `EFG_Matchup` — Shooting efficiency edge
- `TOV_Matchup` — Turnover advantage
- `OREB_Matchup` — Rebounding edge

**Vegas Signals**
- `Vegas_WinProb` — Implied win probability from moneyline
- `Vegas_Spread` — Point spread
- `Vegas_Total` — Over/under total
- `Vegas_WinProb_Edge` — Win probability above 50%

---

## Model

**Algorithm:** Logistic Regression (scikit-learn)
**Hyperparameter tuning:** Azure ML Tune Model Hyperparameters (random sweep, 20 runs)
**Train/test split:** Chronological 80/20 — no random shuffling to prevent future leakage
**Normalization:** Z-Score StandardScaler on all continuous features
**Seasons:** 2013–2024 regular season only

---

## API

The model is deployed as a REST API on Render.com.

**Health check:**
```
GET https://nba-win-predictor-airk.onrender.com/health
```
```json
{"auc": 0.7231, "model": "NBA Win Predictor v2", "status": "healthy"}
```

**Predict:**
```
POST https://nba-win-predictor-airk.onrender.com/predict
Content-Type: application/json

{
  "features": {
    "is_home": 1,
    "Last_5_EFG_PCT": 58.3,
    "Last_5_WinRate": 0.8,
    "Vegas_WinProb": 0.65
  }
}
```
```json
{"win_probability": 0.6842, "lose_probability": 0.3158}
```

> **Note:** Render free tier sleeps after 15 minutes of inactivity. First request may take ~30 seconds to wake up.

---

## Web Application

The web app is a single HTML file — no installation or server required.

**Features:**
- **Matchup Predictor** — Select any two NBA teams, get win probability from the live model
- **Today's Games** — Live NBA schedule from BallDontLie API with model predictions
- **AI Chat** — Ask anything about predictions or the model (powered by Claude API)
- **About Model** — Full technical documentation, feature list, V1 vs V2 comparison

**To use:**
1. Download `nba_ai_predictor_v4.html`
2. Open in Chrome or Safari
3. Add your Claude API key when prompted (get one at console.anthropic.com)
4. The model API connects automatically

---

## Key Findings

**What worked:**
- Switching from raw box scores to Four Factors drove the biggest improvement (+4 AUC points)
- Vegas betting lines added signal unavailable in historical stats (+2 AUC points)
- Pre-engineering all features in a notebook eliminated data type bugs that were corrupting the Designer pipeline

**What didn't work:**
- Spread cover prediction achieved ~0.50 AUC — Vegas spreads are intentionally designed to be unpredictable, and this is the theoretically correct result
- Azure for Students subscriptions lack Container Instance write permissions, preventing direct Azure endpoint deployment

---

## Running Locally

```bash
pip install flask flask-cors scikit-learn numpy joblib gunicorn
python app.py
# API available at http://localhost:5000
```

---

## Course Info

**Course:** Introduction to Artificial Intelligence
**Institution:** University of Detroit Mercy
**Term:** Spring 2026
**Student:** Carlin Percha
