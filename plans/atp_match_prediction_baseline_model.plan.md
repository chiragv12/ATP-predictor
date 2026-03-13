---
name: atp_match_prediction_baseline_model
overview: Build a beginner-friendly Python pipeline that uses open ATP match data (plus surface, conditions, and Elo ratings) to train and evaluate a model that predicts future ATP match winners.
todos:
  - id: env-setup
    content: Set up Python environment and create a starter Jupyter notebook for ATP match prediction work.
    status: pending
  - id: download-data
    content: Download several years of `atp_matches_YYYY.csv` data and place it in a `data/raw` folder, noting the dataset license.
    status: pending
  - id: explore-data
    content: Explore match CSV columns and define initial target and time-based train/validation/test splits in the notebook.
    status: pending
  - id: features-baseline
    content: Implement basic feature engineering (rank differences, surface, round, seeding) and a naive baseline rule.
    status: pending
  - id: train-logreg
    content: Train a logistic regression model with proper preprocessing and evaluate it against the naive baseline on held-out seasons.
    status: pending
  - id: iterate-stronger-models
    content: Experiment with tree-based models and improved features (recent form, optional head-to-head) while maintaining strict time-based evaluation.
    status: pending
  - id: future-preds
    content: Add a notebook section to build feature rows for upcoming matches and generate predicted win probabilities using the trained model.
    status: pending
  - id: elo-system
    content: Implement an Elo rating system (including optional surface-specific Elo) and use pre-match Elo ratings as features and as a baseline predictor.
    status: pending
  - id: weather-features
    content: Join historical weather data (e.g. temperature) to matches by date/location and add temperature-based features to the model.
    status: pending
isProject: false
---

## Goal

**Build an end‑to‑end, beginner‑friendly pipeline in Python that predicts ATP match winners using historical match data (Jeff Sackmann / Tennis Abstract), with proper train/validation/test splits so we can evaluate performance on matches the model hasn’t seen.**

## High-level approach

- **Use public ATP match CSVs** from Jeff Sackmann’s `tennis_atp` repository (the data Tennis Abstract is built on) instead of scraping HTML directly.
- **Work in Python** with `pandas` + `scikit-learn` in a Jupyter notebook so you can experiment interactively.
- **Predict a simple target first**: binary outcome “Did player 1 (the listed winner column) win?” framed as predicting the probability that the eventual winner wins, using only information available *before* the match.
- **Evaluate on future seasons** (e.g. train on 2000–2018, validate on 2019, test on 2020+) to mimic predicting upcoming matches.
- **Exploit tennis-specific factors**: incorporate **surface**, **indoor vs outdoor**, **recent form**, **Elo ratings**, and later **temperature/weather** as key features and baselines.

## Data and environment setup

- **Environment setup**
  - Install Python 3.10+ and create a virtual environment.
  - Install core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and optionally `jupyter`.
  - Create a starter notebook, e.g. `[notebooks/atp_match_baseline.ipynb](notebooks/atp_match_baseline.ipynb)`, where all steps below are implemented.
- **Obtain match data**
  - Either **clone** or **download** selected CSVs from `https://github.com/JeffSackmann/tennis_atp` into a local `data/raw/` folder.
    - Start with tour-level main draw data such as `atp_matches_2000.csv` through `atp_matches_2024.csv`.
  - Optionally also download the player file and ranking files later for extra features.
  - Add a short note about the dataset license (CC BY-NC-SA 4.0) in a `README` section so you remember attribution and non-commercial use.
- **Explore the raw CSVs**
  - Load a single season file in the notebook with `pandas.read_csv`.
  - Inspect columns like `tourney_date`, `surface`, `winner_name`, `loser_name`, `winner_rank`, `loser_rank`, etc.
  - Open `matches_data_dictionary.txt` in the repo (or copy its contents) and skim to understand what each column means.

## Define prediction problem & data split

- **Prediction target**
  - Reframe the data so each row is a *match* with a binary label `y` indicating whether a chosen “focal” player wins.
  - For a first baseline, keep the dataset in the original orientation and set `y = 1` for every row (the existing winner) — the model learns mapping from *pre‑match* features (like rank difference) to the winner. At prediction time for a future match, you’ll construct two records (player A as winner vs player B as winner) and compare their predicted probabilities.
- **Train/validation/test split by time**
  - Add a `season` or `year` column from `tourney_date`.
  - Choose time-based splits, e.g.:
    - Train: 2000–2017
    - Validation: 2018–2019 (for hyperparameter tuning / feature experimentation)
    - Test: 2020–2023 (for final performance estimate)
  - Implement splitting logic that **uses only past years** for training; never mix future matches into training to avoid leakage.

## Feature engineering (first pass)

- **Simple tabular features** from the match CSVs:
  - **Player strength**: `winner_rank`, `loser_rank`, ranking points, and their differences (e.g. `rank_diff = loser_rank - winner_rank`).
  - **Surface and basic conditions**: one-hot encode `surface` (hard, clay, grass, carpet), `tourney_level` (Grand Slams, Masters, ATP 500, etc), and add an `is_indoor` indicator (from an indoor/outdoor flag or derived from tournament info).
  - **Tournament context**: round (`round` column), whether the player is seeded (`winner_seed`, `loser_seed`), best-of-3 vs best-of-5 (can be inferred from `tourney_level` and Slam flags).
- **Recent form features** (simple moving stats):
  - Sort matches by date and, for each player, compute:
    - Rolling win rate over last N matches (e.g. 10 and 25).
    - Count of matches played in last 30/60 days (fatigue/fitness proxy).
  - Implement this via grouped operations: group by player ID, sort by date, and use rolling windows. Ensure that for each match, recent form is computed *only from matches strictly before that match date*.
- **Head-to-head features (optional in first pass)**
  - For each pair of players, compute historical head-to-head record prior to the match date (wins/losses for each player).
  - Start without H2H to keep it simple; add later once the baseline works.
- **Feature matrix**
  - Assemble a `pandas` DataFrame with numeric and one-hot encoded features plus the target `y`.
  - Use `scikit-learn`’s `ColumnTransformer` + `Pipeline` to handle preprocessing (e.g. `OneHotEncoder` for categoricals and `StandardScaler` for numeric features) so the transformations are reproducible.

## Elo rating system and Elo-based features

- **Global Elo ratings**
  - Sort all matches chronologically by `tourney_date`.
  - Initialize each player with a base Elo (e.g. 1500).
  - For each match in time order:
    - Compute expected win probabilities from current Elo ratings.
    - Update ratings using a standard Elo update with a chosen `K` factor.
  - Store **pre-match Elo** for each player in that row (these become features).
- **Surface-specific Elo (optional extension)**
  - Maintain separate Elo ratings per surface (e.g. `elo_hard`, `elo_clay`, `elo_grass`) and update only the relevant surface Elo for each match.
  - Add differences like `elo_diff`, `elo_clay_diff`, etc., as features when appropriate.

## Weather and temperature features (phase 2)

- **Historical weather join**
  - For each match, use tournament location and `tourney_date` to query a historical weather API (e.g. Meteostat/Visual Crossing) and obtain match-day **temperature** (and optionally humidity).
  - Build a lookup table keyed by `(location, date)` and merge weather into the match dataset.
- **Weather-derived features**
  - Add numeric `temp_c` and/or simple categories (e.g. cold/normal/hot) plus optional flags like `hot_day` if `temp_c` exceeds a threshold.
  - Retrain models with and without weather features to measure the added value.

## Baseline modeling

- **Establish a very simple baseline first**
  - Compute a **naive “always pick higher-ranked player” rule** and measure its accuracy on the test set.
  - This gives you a benchmark that any ML model should beat.
- **Train a logistic regression model**
  - Use `LogisticRegression` from `scikit-learn` wrapped inside a `Pipeline` that includes preprocessing.
  - Train on the training seasons and tune simple hyperparameters (e.g. regularization strength `C`) using the validation seasons.
  - Evaluate on the held‑out test seasons using:
    - Accuracy and balanced accuracy.
    - Log-loss or Brier score (because we care about calibrated probabilities).
    - Optional: calibration curve plots.
- **Compare vs baseline**
  - Compare the logistic regression performance to the naive rule.
  - Inspect feature coefficients to see which factors seem most important (e.g. rank diff, surface, round).

## Stronger models (after baseline works)

- **Tree-based models**
  - Try models like `RandomForestClassifier` or `GradientBoostingClassifier` / `XGBoost` / `LightGBM` on the same feature set.
  - Use the same train/validation/test time split; avoid cross-validation that shuffles across years (to respect temporal structure).
  - Compare performance metrics to logistic regression.
- **Probability calibration**
  - If a tree model performs better but gives overconfident probabilities, wrap it in `CalibratedClassifierCV` or use isotonic regression / Platt scaling trained on the validation seasons.

## Using the model for future matches

- **Building feature rows for upcoming matches**
  - For a future date (e.g. upcoming tournament), construct the feature vector for each possible matchup using the **latest available historical data** up to that date.
  - Use current rankings, surfaces, and recent form computed from prior matches only.
  - For each match, construct two records (A beats B, B beats A) or define a consistent “player1/player2” orientation and interpret probabilities accordingly.
- **Generating predictions**
  - Load the trained model and preprocessing pipeline (persisted via `joblib` or `pickle`).
  - Apply to the upcoming matches feature matrix and output predicted win probabilities.
  - Save to CSV or display nicely in the notebook (e.g. table with player names, tournament, surface, model probability).

## Guardrails and learning focus

- **Avoid data leakage**
  - Be strict that all features used for a match are derived only from information available before the match date.
  - Keep ranking and recent-form computations forward-looking only, using cumulative/rolling operations.
- **Iterative learning**
  - Start with:
    - 1–2 seasons of data,
    - a small feature set (rank diff + surface + round),
    - one simple model (logistic regression).
  - Once you’re comfortable, scale to more years, add more features, and test stronger models.
- **Documentation and reproducibility**
  - Keep all steps (data loading, cleaning, feature engineering, model training, evaluation) in the notebook in clearly separated sections.
  - Optionally, factor stable code (like feature engineering functions) into small Python modules under `src/` later for reuse.

