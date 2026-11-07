# Elo ratings

Code: [`src/elo.py`](../src/elo.py) · Notebook: [`02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)

## The idea

Every player has one number, their **rating**. Before a match, the gap between the two ratings
gives a predicted win probability. After the match, each player's rating moves by how **surprising**
the result was: beating someone you were expected to beat earns a little, an upset earns a lot.

Nobody tells Elo who is good. It learns that purely from who beats whom, which makes it a
from-scratch alternative to the official ATP ranking (which is points earned at tournaments in the last 52 weeks).

## Step 1 — predicted win probability

```
P(A beats B) = 1 / (1 + 10 ** ((elo_B - elo_A) / 400))
```

Only the **difference** matters, not the absolute values:

| Rating gap (A − B) | P(A wins) |
|---:|---:|
| 0 | 50% |
| +100 | 64% |
| +200 | 76% |
| +400 | 91% |

The 400 is a scale constant: every 400 points of gap multiplies the odds by 10.
In code: `elo_win_probability(elo_a, elo_b)`.

## Step 2 — update after the match

```
new_elo = old_elo + K * (actual - expected)
```

- `actual` = 1 for the winner, 0 for the loser
- `expected` = the probability from step 1
- `K` = how big a step to take

The winner always gains and the loser always loses. How much depends on how surprised Elo was:
a 90% favourite who wins gains only `K × 0.1`; a 10% underdog who wins gains `K × 0.9`.

## Step 3 — K: how fast ratings move

A fixed K forces a trade-off: big K reacts quickly but is noisy, small K is stable but slow to notice
a rising newcomer. We use FiveThirtyEight's tennis rule, where K shrinks as a player plays more matches:

```
K = 250 / (matches_played + 5) ** 0.4
```

| Matches played | 0 | 10 | 50 | 100 | 300 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| K | 131 | 85 | 50 | 39 | 25 | 19 |

A newcomer's rating finds its level within a few dozen matches; a veteran's barely moves on one result.
Because each player has their own K, the winner's gain and the loser's loss aren't always equal.

## Worked example — 2024 Australian Open final

Jannik Sinner beat Daniil Medvedev 3-6 3-6 6-4 6-4 6-3. Ratings going in (from this project's data):

| | Sinner | Medvedev |
|---|---:|---:|
| Pre-match Elo | 2222.3 | 2136.6 |
| Matches played before | 270 | 473 |
| K | 26.4 | 21.2 |

1. **Expected:** gap = 2222.3 − 2136.6 = 85.7 → `1 / (1 + 10 ** (−85.7/400))` = **0.621**. Sinner was a 62% favourite.
2. **Sinner (won):** 2222.3 + 26.4 × (1 − 0.621) = 2222.3 + **10.0** = **2232.3**
3. **Medvedev (lost):** 2136.6 + 21.2 × (0 − 0.379) = 2136.6 − **8.0** = **2128.6**

The code agrees: Sinner's next match (Rotterdam) starts at 2232.3 and Medvedev's (Dubai) at 2128.6.

Compare an **upset**: in the 2025 Roland Garros final Alcaraz (2221) beat Sinner (2335), with Elo giving Alcaraz only
34%. Same formula, but now the winner's multiplier is 0.66 instead of 0.38: Alcaraz (K = 25.2) gained **16.6** points,
versus Sinner's 10.0 for his expected win above.

## A real trajectory — Jannik Sinner

| Year | Matches | Elo at start | Peak | Elo at end |
|---|---:|---:|---:|---:|
| 2019 | 21 | 1500 | 1741 | 1741 |
| 2020 | 31 | 1792 | 1951 | 1951 |
| 2021 | 72 | 1969 | 2034 | 1976 |
| 2022 | 64 | 1988 | 2076 | 2008 |
| 2023 | 84 | 1982 | 2185 | 2185 |
| 2024 | 81 | 2190 | 2311 | 2311 |
| 2025 | 64 | 2312 | 2339 | 2333 |
| 2026* | 40 | 2343 | 2375 | 2375 |

\* through Roland Garros. All values are **pre-match** ratings, so "end" is the rating going into his last match of the year;
the next year's "start" already includes that last result (e.g. 2019 → 2020 jumps 1741 → 1792 because he won his final 2019 match).

- **2019:** debut at the default 1500 with a huge K, so his first season moves ~240 points.
- **2021–22:** plateau around 2000 — a good top-20 player.
- **2023–24:** the breakthrough. +300 over two seasons, ending 2024 as the top-rated player.
- **2025–26:** small changes despite winning big titles — he's expected to win now, and K is small after 400+ matches.

As of May 2026 the top of the table is Sinner (2375), Alcaraz (2283), Djokovic (2171) — sensible.
Federer and Nadal also still appear high: a retired player's rating simply freezes at their last match.

## Surface Elo

`compute_surface_elo_ratings` runs the exact same algorithm but keeps a **separate rating per (player, surface)**,
updated only by matches on that surface. A clay specialist can then be 2100 on clay and 1800 on grass.
Grass has the fewest matches (~10% of the data), so grass ratings are the noisiest.

## Why it's leakage-safe

The code walks through matches in chronological order and, for each one:

1. **reads** both players' current ratings and stores them as that match's feature, *then*
2. **updates** the ratings using the result.

So a match's feature can never contain its own result or any later one. Notebook 02 proves it: features built
from data cut off at 2019 are identical to the same rows built from the full 2010–2026 data.

Walkovers record a pre-match rating but don't update anything — no tennis was played.

## What we found on this data

Picking the higher-Elo player is right about as often as picking the better-ranked player (~65%), not clearly better.
Two reasons:

- **Cold start.** Our data begins in 2010, so in 2010 Djokovic and Nadal start at 1500 like everyone else.
  Elo is weakest that first year (65% vs rank's 68%) and catches up by 2012 — Djokovic, already world #3, doesn't reach his true rating until mid-2011. Later phases can skip the first season or two as "burn-in".
- **Tour-level matches only.** The dataset has no Challenger or qualifying events, so lower-ranked players
  have few matches to learn from, while the official ranking does count those events.

Elo's value comes later: combined with rank, form and head-to-head in a model, it adds information the ranking misses
(e.g. who a player has *beaten*, not just how many points they've earned).
