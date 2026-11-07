"""Leakage-safe pre-match features and the model-ready feature table.

Every feature for a match uses only matches that came BEFORE it (rows are in chronological order).
"""
import numpy as np
import pandas as pd

from elo import compute_elo_ratings, compute_surface_elo_ratings

UNRANKED = 2000  # stand-in rank for unranked players (see notebook 01, section 4)


def add_recent_form(df, n=10):
    """winner_form / loser_form: win rate over each player's previous n played matches (NaN if none yet).

    Walkovers are left out of the history and get NaN themselves.
    """
    played = df[~df["walkover"]]
    # Long format: two rows per match, one per player. "order" keeps the chronological position.
    long = pd.concat([
        pd.DataFrame({"order": played.index, "player_id": played["winner_id"], "won": 1.0, "side": "winner"}),
        pd.DataFrame({"order": played.index, "player_id": played["loser_id"], "won": 0.0, "side": "loser"}),
    ]).sort_values("order", kind="stable")
    # shift(1) drops the current match from its own window -- that's what prevents leakage.
    long["form"] = long.groupby("player_id")["won"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    form = long.pivot(index="order", columns="side", values="form")
    return df.assign(winner_form=form["winner"], loser_form=form["loser"])


def add_head_to_head(df):
    """winner_h2h / loser_h2h: wins each player had against this same opponent before this match."""
    lo = np.minimum(df["winner_id"], df["loser_id"])
    hi = np.maximum(df["winner_id"], df["loser_id"])
    pair = lo.astype(str) + "-" + hi.astype(str)
    lo_won = ((df["winner_id"] == lo) & ~df["walkover"]).astype(int)
    hi_won = ((df["winner_id"] == hi) & ~df["walkover"]).astype(int)
    # cumsum includes this match; subtract it back out so only earlier meetings count.
    lo_prior = lo_won.groupby(pair).cumsum() - lo_won
    hi_prior = hi_won.groupby(pair).cumsum() - hi_won
    winner_is_lo = df["winner_id"] == lo
    return df.assign(
        winner_h2h=np.where(winner_is_lo, lo_prior, hi_prior),
        loser_h2h=np.where(winner_is_lo, hi_prior, lo_prior),
    )


def build_feature_table(df):
    """One row per match in a random player_1/player_2 orientation, label p1_won, features as p1 - p2.

    df: output of load_matches() (chronological). Includes walkovers/retirements, flagged, so
    models can choose what to train on.
    """
    df = pd.concat([df, compute_elo_ratings(df), compute_surface_elo_ratings(df)], axis=1)
    df = add_head_to_head(add_recent_form(df))
    pts = ["winner_rank_points", "loser_rank_points"]
    df[pts] = df[pts].fillna(0)  # unranked = no ranking points

    # Deterministic coin flip per match (hash of its id), so rebuilding gives the same orientation.
    flip = (pd.util.hash_pandas_object(df[["tourney_id", "match_num"]], index=False) % 2 == 1).to_numpy()

    def side(winner_col, loser_col):
        """(player_1 values, player_2 values) after the flip."""
        w, l = df[winner_col].to_numpy(), df[loser_col].to_numpy()
        return np.where(flip, l, w), np.where(flip, w, l)

    out = df[["tourney_id", "tourney_name", "tourney_date", "match_num", "surface", "tourney_level",
              "round", "best_of", "walkover", "retired"]].copy()
    out["p1_id"], out["p2_id"] = side("winner_id", "loser_id")
    out["p1_name"], out["p2_name"] = side("winner_name", "loser_name")
    out["p1_won"] = (~flip).astype(int)

    p1_rank, p2_rank = side("winner_rank", "loser_rank")
    out["p1_rank"] = pd.Series(p1_rank, index=df.index).fillna(UNRANKED)
    out["p2_rank"] = pd.Series(p2_rank, index=df.index).fillna(UNRANKED)
    # Log, because the gap between #1 and #10 matters far more than #201 vs #210. Positive = p1 ranked WORSE.
    out["log_rank_diff"] = np.log(out["p1_rank"]) - np.log(out["p2_rank"])

    for name, (w, l) in {
        "rank_points_diff": ("winner_rank_points", "loser_rank_points"),
        "elo_diff": ("winner_elo", "loser_elo"),
        "surface_elo_diff": ("winner_surface_elo", "loser_surface_elo"),
        "form_diff": ("winner_form", "loser_form"),
        "h2h_diff": ("winner_h2h", "loser_h2h"),
    }.items():
        p1, p2 = side(w, l)
        out[name] = p1.astype(float) - p2.astype(float)
    return out
