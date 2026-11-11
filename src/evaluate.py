"""Train/validation split, baselines, and metrics shared by every model (Phases 3-6)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from elo import elo_win_probability

TRAIN_START = "2012-01-01"  # 2010-11 only warm up Elo (cold start, notebook 02 section 7)
VAL_START = "2025-04-01"    # last 9 months of 2025: for comparing models and tuning
TEST_START = "2026-01-01"   # 2026 is the held-out test (Phase 6) -- never touched here


def split(ft):
    """(train, val) date slices of the feature table, keeping only matches played to a finish.

    Walkovers never happened and retirements are partly injuries, so neither is a fair label.
    They still count toward Elo/form/H2H -- that already happened when the table was built.
    """
    played = ft[~ft["walkover"] & ~ft["retired"]]
    d = played["tourney_date"]
    train = played[(d >= TRAIN_START) & (d < VAL_START)]
    val = played[(d >= VAL_START) & (d < TEST_START)]
    return train, val


def naive_baseline_predict(train, frame):
    """P(p1 wins) from ranking alone: the better-ranked player wins at the rate seen in `train`.

    A fixed rate instead of 0/1 so log-loss is finite (a 100%-confident wrong guess costs infinity).
    """
    fav_rate = ((train["p1_rank"] < train["p2_rank"]) == (train["p1_won"] == 1)).mean()
    p1_better = frame["p1_rank"] < frame["p2_rank"]
    p1_worse = frame["p1_rank"] > frame["p2_rank"]
    return np.where(p1_better, fav_rate, np.where(p1_worse, 1 - fav_rate, 0.5))


def elo_baseline_predict(frame):
    """P(p1 wins) straight from the Elo formula -- no training at all."""
    return elo_win_probability(frame["elo_diff"].to_numpy(), 0)


def evaluate_predictions(y_true, p1_prob):
    """Accuracy (picked the right player?), log-loss and Brier (how good were the probabilities?)."""
    return {
        "accuracy": accuracy_score(y_true, np.asarray(p1_prob) > 0.5),
        "log_loss": log_loss(y_true, p1_prob),
        "brier": brier_score_loss(y_true, p1_prob),
    }


def accuracy_gap(y_true, p_a, p_b):
    """How much more often A picks the winner than B, with a 95% margin of error (paired, same matches).

    If the margin is bigger than the gap, the validation set can't tell the two apart.
    """
    y = np.asarray(y_true) == 1
    diff = ((np.asarray(p_a) > 0.5) == y).astype(int) - ((np.asarray(p_b) > 0.5) == y).astype(int)
    return diff.mean(), 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))


def summarize_results(results):
    """{model name: evaluate_predictions(...)} -> one comparison row per model."""
    return pd.DataFrame(results).T.round(4)
