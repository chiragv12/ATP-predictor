"""Elo ratings computed from scratch in one chronological pass. See docs/01_elo.md."""
import pandas as pd

START = 1500


def elo_win_probability(elo_a, elo_b):
    """Expected score (win probability) of A against B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def k_factor(matches_played):
    """Big steps for newcomers, small steps for veterans (FiveThirtyEight's tennis K)."""
    return 250 / (matches_played + 5) ** 0.4


def _elo_pass(df, winner_keys, loser_keys):
    """Record each player's rating BEFORE the match, then update. Rows must be in chronological order.

    Walkovers get pre-match ratings but don't update anything (no tennis was played).
    A key of None (e.g. unknown surface) yields NaN and no update.
    """
    rating, played = {}, {}
    w_pre, l_pre = [], []
    for w, l, walkover in zip(winner_keys, loser_keys, df["walkover"]):
        if w is None or l is None:
            w_pre.append(float("nan"))
            l_pre.append(float("nan"))
            continue
        rw, rl = rating.get(w, START), rating.get(l, START)
        w_pre.append(rw)
        l_pre.append(rl)
        if walkover:
            continue
        expected_w = elo_win_probability(rw, rl)
        rating[w] = rw + k_factor(played.get(w, 0)) * (1 - expected_w)
        rating[l] = rl - k_factor(played.get(l, 0)) * (1 - expected_w)
        played[w] = played.get(w, 0) + 1
        played[l] = played.get(l, 0) + 1
    return pd.DataFrame({"winner_elo": w_pre, "loser_elo": l_pre}, index=df.index)


def compute_elo_ratings(df):
    """Pre-match overall Elo for winner and loser of every match."""
    return _elo_pass(df, df["winner_id"], df["loser_id"])


def compute_surface_elo_ratings(df):
    """Pre-match Elo counting only matches on the same surface."""
    # Carpet (66 matches, all Davis Cup ties) is too rare for its own rating; it's a fast indoor court, so count it as Hard.
    surface = df["surface"].replace({"Carpet": "Hard"})
    known = surface.notna()
    w = [(p, s) if k else None for p, s, k in zip(df["winner_id"], surface, known)]
    l = [(p, s) if k else None for p, s, k in zip(df["loser_id"], surface, known)]
    return _elo_pass(df, w, l).rename(columns=lambda c: c.replace("_elo", "_surface_elo"))
