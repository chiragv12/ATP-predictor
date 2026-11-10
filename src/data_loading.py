"""Load Jeff Sackmann's ATP match CSVs (data/raw/atp_matches_YYYY.csv, atp_players.csv)."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
# Play order within a tournament. RR = round robin (group stage, before knockouts); BR = Olympic bronze match (before the final).
ROUND_ORDER = {"RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "BR": 7, "F": 8}


def load_matches(raw_dir=RAW_DIR):
    """All years concatenated, in chronological order, with incomplete matches flagged (not dropped)."""
    files = sorted(Path(raw_dir).glob("atp_matches_[0-9][0-9][0-9][0-9].csv"))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv files in {raw_dir} -- see README")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    # tourney_date is the tournament's START date, shared by every match in the event. match_num alone
    # isn't always play order (~19 events list the final first), so sort by round, then match_num.
    df["round_order"] = df["round"].map(ROUND_ORDER)
    assert df["round_order"].notna().all(), f"unknown round: {set(df['round']) - set(ROUND_ORDER)}"
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"]).reset_index(drop=True)
    assert df.groupby("tourney_id")["round_order"].is_monotonic_increasing.all()
    # Elo/form need every match; models can filter on these later.
    score = df["score"].fillna("")
    # Spelling varies across years: "W/O"/"Walkover", "RET"/"DEF"/"Def."/"Default"/"ABD"/"Played and abandoned".
    df["walkover"] = score.str.contains("W/O|walkover", case=False)
    df["retired"] = score.str.contains("RET|DEF|ABD|abandoned", case=False)
    return df


def load_players(raw_dir=RAW_DIR):
    players = pd.read_csv(Path(raw_dir) / "atp_players.csv", dtype={"dob": str, "wikidata_id": str})
    # Some dobs are missing or partial (e.g. day 00) -> NaT.
    players["dob"] = pd.to_datetime(players["dob"], format="%Y%m%d", errors="coerce")
    return players
