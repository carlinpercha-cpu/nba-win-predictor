# EDGE Sports Intelligence

**A multi-sport win-probability system, and why its offline results were never validated in production**

Carlin Percha · University of Detroit Mercy · MS Applied Data Analytics
Live application: <https://carlinpercha-cpu.github.io/nba-win-predictor/>
Model API: <https://nba-win-predictor-airk.onrender.com/health>

---

## Summary

This project trains binary win-probability models for eight sports and serves them
through a web application with live odds, LLM-generated analysis, and multi-book
line shopping. Held-out AUC ranges from 0.617 (MLB) to 0.867 (CFB) on chronological
splits.

Two things must be said about those numbers.

**First, they do not transfer to production, and this is demonstrable.** The serving
client supplies a fraction of each model's trained feature vector — seven of
fifty-three for NBA — and the API zero-fills the remainder without flagging it.
Direct probes confirm the consequence: the NBA model returns 23.1% for a home side
the market prices at 85%, and the tennis model returns 46.3% for an all-neutral
matchup regardless of the market price passed to it. This is train/serve skew, not
model error, and it is reproducible on demand.

**Second, there is no live performance figure, because the evaluation infrastructure
was never validated before it was relied upon.** Prediction logging to Google Sheets
dropped writes silently under memory pressure; the scheduled jobs that resolved
outcomes and captured closing lines failed and were not monitored; the free-tier
service was ultimately unable to sustain the workload and was disabled. Rows exist
in the sheet, but their completeness is unknown and their coverage is intermittent.
No accuracy or closing-line-value claim can be made from them, and none is made here.

An earlier version of this README reported ~68% accuracy and compared it favourably
to professional oddsmakers. That was an offline test-set number presented as though
it described live performance. It has been removed.

What this project actually establishes is documented below: eight offline models, a
mechanistically demonstrated serving failure, a negative result on rebuilding the
weakest model, and a methodological lesson about instrumenting evaluation before
trusting it.

---

## System

```
Historical data ──► feature engineering ──► scikit-learn models (8 sports)
                                                    │
                                                    ▼
GitHub Pages (client) ◄──── Flask API on Render ────┴──► Google Sheets (intended log)
        │                          │
        │                          ├──► The Odds API (live lines, 8+ books)
        │                          └──► Anthropic API (scouting reports, analyst chat)
        │
        └──► ESPN / BallDontLie (schedules, scores)
```

**Serving.** Flask on Render's free tier: 512 MB, single gunicorn worker, sleeps
after fifteen minutes idle. Models are lazily loaded with LRU eviction capped at
three resident models. Before that cap was added the process was OOM-killed several
times daily.

**Client.** A single static HTML file on GitHub Pages. Confidence tiering by edge
over the market (HIGH ≥ 8 pp, MED ≥ 4 pp), best-price line shopping across books,
a bet tracker with ROI, and LLM-generated scouting reports. These components work
and are independent of the prediction pipeline.

**Intended evaluation, not achieved.** The design called for every prediction to be
written to a sheet with game ID, both implied probabilities, the pick, and the market
price at prediction time; a scheduled job to resolve outcomes from final scores; and
a second job to capture closing lines for CLV. Each piece was built. None was
instrumented well enough to detect its own failure, and all three failed. See
*Infrastructure failure* below.

---

## Models

All models are scikit-learn (logistic regression or random forest), isotonically
calibrated via `CalibratedClassifierCV`, and evaluated on a chronological 80/20
split — no random shuffling, so no future leakage into the training window.

| Sport | Held-out AUC | Test accuracy | Features | Training set | Seasons |
|---|---|---|---|---|---|
| College Football | 0.867 | 77.6% | 26 | 4,444 games | 2019–2024 |
| College Basketball | 0.862 | 77.4% | 38 | 37,635 games | 2013–2024 |
| NFL | 0.788 | 66.0% | 42 | 1,087 games | 2021–2024 |
| NHL | 0.749 | 67.5% | 27 | 22,631 games | 2010–2026 |
| NBA | 0.727 | 66.5% | 53 | 24,290 games | 2013–2025 |
| EPL | 0.707 | 57.9% (3-way) | 26 | 3,704 matches | 2015–2025 |
| ATP Tennis | 0.695 | 64.0% | 22 | 24,521 matches | 2015–2024 |
| MLB | 0.617 | 58.1% | 20 | 2,305 games | pitcher-driven |

A separate neutral-site model for single-elimination NCAA tournament matchups reaches
0.927 AUC on 1.75M matchup pairs; it does not run on the daily slate and its number
is not comparable to the others.

Every figure above was measured on a complete feature vector. The following section
explains why that matters more than it should.

---

## Finding: train/serve feature skew

Two direct API probes isolate the failure. Neither depends on the prediction log.

**NBA.** Queried with a decisive home favourite: market-implied probability 0.85,
superior offensive and defensive ratings, better recent form, a rest advantage,
opponent on a back-to-back and a losing streak.

```
POST /predict  {"sport":"nba","vegas_prob":0.85, ...}
→ {"win_probability": 0.231, "features_used": 53, "missing_features": []}
```

23.1% for a team the market prices at 85%. Note `missing_features: []` — the endpoint
reports nothing missing because absent keys are silently zero-filled rather than
flagged. `/features/nba` shows the model expects `Last_5_OFFRTG`, `Last_10_DEFRTG`,
`Last_5_PIE`, `Last_5_PTS_OFF_TO`, `Vegas_WinProb`, and forty-eight others. The
client's payload builder supplies seven. The remaining forty-six arrive as zero, and
a team with a zeroed offensive rating sits far outside anything in the training
distribution.

**Tennis.** Queried with entirely neutral features and a market price of 0.60:

```
POST /predict  {"sport":"tennis", ...all diffs 0..., "vegas_prob":0.60}
→ {"win_probability": 0.463}
```

Two faults compound. The tennis model was trained with `player_a` assigned at random
between winner and loser, so it encodes no notion of favourite — while the client
assigns the market favourite to `player_a` and expects that to be honoured. And
`vegas_prob` was never a tennis training feature, so passing it is inert. The model
returns its response to an all-neutral vector, every time, for every match.

**Why this went unnoticed.** `missing_features: []` is itself the bug. Zero-filling
converts a loud, catchable failure into a silent one. An endpoint that reported which
features were imputed would have surfaced this on the first request rather than
requiring a manual probe months later.

The offline AUC table is therefore an upper bound on a configuration production never
reached — not a description of anything the deployed system did.

---

## Infrastructure failure

Recorded in full because it is the reason no live result exists.

**Memory.** Render's free tier allots 512 MB. Default gunicorn spawns multiple
workers, each loading its own copy of every model touched. The service was OOM-killed
repeatedly. Pinning to a single worker and capping resident models at three via LRU
eviction reduced but did not eliminate this.

**Silent write loss.** Sheets writes were wrapped in exception handling that logged
to stdout and continued. Under memory pressure and cold starts these calls timed out,
and each timeout dropped a prediction with no visible effect on the client. There was
no write-confirmation check and no row-count reconciliation.

**Unmonitored schedulers.** Outcome resolution, closing-line capture, and daily
team-stat refresh ran as external cron jobs against the API. When they began failing
there was no alerting, and their failure is invisible in the application — a sheet
with stale rows looks identical to a sheet with no games to resolve.

**Termination.** The service was ultimately disabled for exceeding free-tier
resources.

**Net effect.** The prediction sheet contains rows, but with unknown drop rate,
intermittent coverage, and partial outcome resolution. It cannot support an accuracy
estimate, a calibration curve, or a CLV figure. The correct treatment is to discard
it rather than to report a number with caveats attached, and that is what has been
done here.

---

## Negative results

Recorded rather than omitted, because each narrowed the search.

**MLB rebuild without market prices (abandoned).** The MLB model is the weakest at
0.617 AUC and trained on 2017–2019, so a rebuild was attempted on 2019–2024.
Baseball Reference schedules yielded 8,853 games; Baseball Savant yielded 1.43M
pitches, aggregated to team-game wOBA, launch speed, home runs, and strikeout rate,
and merged onto 4,269 games. Rolling 5- and 10-game windows were computed throughout.

The result was **0.5088 AUC** — worse than the model it was meant to replace, and
barely above chance. Coefficient inspection showed `team_woba_l10` (1.07) and
`team_woba_l5` (0.62) dominating with `is_home` at 0.12: the model had learned recent
offensive form and essentially nothing else. The cause is the omission of market
prices. In the incumbent model the market-implied probability is the load-bearing
feature, and statcast form does not substitute for it. This is consistent with the
general pattern that in efficient markets, structural features are weak substitutes
for the price.

**Signed streak (kept, but not the fix).** The original streak feature counted only
consecutive wins and floored at zero, discarding every losing streak — 4,325 of 8,853
team-games carried no information. Replacing it with a signed streak spanning −12 to
+13 moved AUC from 0.5088 to 0.5107. Correct, worth keeping, and a rounding error
against the missing-price problem.

**Historical odds recovery (partial).** Sportsbookreviewsonline archives yielded
10,751 MLB games with opening and closing moneylines, parsed into game pairs with
implied probabilities. Coverage stops at 2021, so it cannot support a 2022–2024
rebuild. FanGraphs endpoints returned HTTP 403 to programmatic requests. BallDontLie
moved both its advanced-stats and season-averages endpoints behind a paid tier during
the project, removing the most direct route to the NBA features the model needs.

**Spread-cover prediction (prior work, unchanged).** Predicting against the spread
reached ~0.50 AUC. This is the theoretically correct result — spreads are set to
split action — and is reported as evidence the evaluation harness detects a genuine
null.

---

## Data sources

| Domain | Source | Role |
|---|---|---|
| NBA | Kaggle traditional + four-factors + betting sets | Training |
| MLB | Baseball Reference (`pybaseball`), Baseball Savant statcast | Training |
| MLB odds | Sportsbookreviewsonline archives, 2017–2021 | Training |
| EPL | football-data.co.uk | Training |
| Tennis | Jeff Sackmann `tennis_atp` | Training |
| Schedules, scores | ESPN public API, BallDontLie | Serving |
| Live odds | The Odds API, multi-book | Serving |
| Analysis | Anthropic API | Serving |

---

## Limitations

**No live evaluation exists.** Not a small sample — no valid sample. Every claim in
this document is either an offline measurement or a direct API probe.

**Offline AUCs describe an unreached configuration.** They were measured on complete
feature vectors that the serving path does not construct.

**Feature availability was never a design constraint.** Models were trained on the
richest available historical data without first checking whether equivalent features
could be obtained daily, freely, at inference time. Advanced NBA metrics in
particular are gated behind paid APIs, and that gating tightened mid-project.

**Free-tier hosting was not viable for this workload.** Eight models, scheduled jobs,
and a persistent log exceed what 512 MB and a sleeping instance can support. This is
a design mismatch, not a tuning problem.

**Market efficiency bounds the ceiling regardless.** Closing lines aggregate injury
news, lineup information, and sharp money that none of these feature sets contain.
Beating a close is materially harder than predicting a winner, and CLV — not
accuracy — is the metric that would demonstrate it.

---

## Next steps

In order, with the criterion that would kill each.

1. **Instrument before measuring.** Have `/predict` return the names of zero-filled
   features rather than an empty `missing_features` list, and log the fill rate per
   request. Verify every log write with a read-back before treating the log as data.
   Nothing else on this list is worth doing first.
2. **Retrain to the serving schema.** Restrict each model to features obtainable
   daily from free sources and treat the resulting AUC as the real number.
   *Kill criterion:* if a schema-restricted model cannot beat a market-price-only
   baseline on held-out data, it should not be served at all.
3. **Fix hosting or reduce scope.** Either move to a tier that can hold the workload,
   or cut to two or three sports that can run within the constraint. Continuing to
   run eight models on a free instance guarantees a repeat.
4. **Pre-commit an evaluation window.** Log to a fixed, pre-registered number of
   resolved games before drawing any conclusion, rather than reading a rough week as
   a regime change.
5. **Report CLV as the primary metric.** Accuracy against a market-adjacent baseline
   is close to uninformative. Whether predictions beat the close is the question.

---

## Running locally

```bash
pip install flask flask-cors scikit-learn numpy joblib gunicorn
python app.py            # API on http://localhost:5000
```

Model artefacts are pinned to scikit-learn 1.5.2 to match the deployment
environment. The client is a single HTML file and requires no build step.

---

## Provenance

Originally submitted as a single-sport NBA project for Introduction to Artificial
Intelligence, University of Detroit Mercy, Spring 2026. Subsequently extended to
eight sports with line shopping, an LLM analyst interface, and a prediction-logging
pipeline that did not work. This revision replaces the original results section,
which reported offline accuracy as though it described live performance, and removes
a live accuracy figure derived from an unvalidated log.
