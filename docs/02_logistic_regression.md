# Logistic regression

Code: [`src/evaluate.py`](../src/evaluate.py) · Notebook: [`03_baseline_and_logreg.ipynb`](../notebooks/03_baseline_and_logreg.ipynb)

## The idea

We want a **probability** that player 1 wins, from the feature differences (`elo_diff`, `log_rank_diff`, ...).
Logistic regression does it in two steps:

1. **Score:** a weighted sum of the features — `z = w₁·x₁ + w₂·x₂ + … + w₆·x₆`
2. **Squash:** turn the score into a probability with the **sigmoid** — `p = 1 / (1 + e^(−z))`

| z | −3 | −1 | 0 | +1 | +3 |
|---|---:|---:|---:|---:|---:|
| p | 5% | 27% | 50% | 73% | 95% |

Any score maps to something between 0 and 1. A positive score favours player 1; zero is a coin flip.

**Training** = finding the weights `w` that make the probabilities match what actually happened as closely as possible
(details below). That's the whole model — six numbers.

### You've already met one

Elo's win formula, `1 / (1 + 10^(−elo_diff/400))`, is a sigmoid too, with one feature and a weight fixed by hand
(`ln(10)/400`). Logistic regression is the same shape, but it **learns** the weights and combines several features.

## Reading the weights: log-odds

`z` is the **log-odds** of player 1 winning: `z = ln(p / (1 − p))`. So each weight says how much one feature moves the
log-odds, holding the others fixed. `e^w` is easier to read — it's an **odds multiplier**:

- `w = +0.26` → `e^0.26 = 1.30`: each step up in that feature multiplies p1's odds by 1.30 (+30%)
- `w = −0.37` → `e^−0.37 = 0.69`: each step multiplies the odds by 0.69 (−31%)

What's "one step"? We **standardize** every feature first (subtract its mean, divide by its standard deviation), so a step is
**one standard deviation**: ~212 Elo points for `elo_diff`, ~1.5 for `log_rank_diff`. That puts all weights on the same scale,
so you can compare them directly — and it makes regularization (below) treat every feature fairly.

## Training: log-loss

Training minimises **log-loss**, the average of `−ln(probability you gave to what actually happened)`:

| You said P(p1 wins) | p1 won | p1 lost |
|---:|---:|---:|
| 0.5 | 0.69 | 0.69 |
| 0.7 | 0.36 | 1.20 |
| 0.9 | 0.11 | **2.30** |
| 0.99 | 0.01 | **4.61** |

Being right earns little; being **confidently wrong** is punished hard. So log-loss rewards *honest* probabilities, not just
picking the right player. That's why it's our main metric alongside accuracy (lower is better; always saying 0.5 scores 0.693).
**Brier score** is a gentler cousin: the average of `(p − outcome)²`.

## Regularization: C

Left alone, the model can chase noise in the training data with large weights. Regularization adds a penalty for big weights;
in scikit-learn `C` controls it — **smaller C = stronger penalty = weights pulled towards 0**. We pick C by trying several values
and keeping whichever does best on the **validation** set (data the model wasn't trained on).

Here it barely matters (validation log-loss 0.610–0.612 across C = 0.001 … 1), because 36,000 training matches and six features
leave little room to overfit. We use C = 0.001.

## Design choices for tennis

**The split.** Train on 2012 → Mar 2025 (36,193 matches), validate on Apr → Dec 2025 (1,872). 2010–11 only warm up Elo.
Walkovers and retirements are left out of both. 2026 is untouched until Phase 6.

**Only the six differences — no intercept, no surface/round/level columns.** Because player 1 is chosen by coin flip, *who is
player 1* carries no information. A model is fair only if swapping the players flips its answer (`p → 1 − p`). With difference
features and no intercept that's guaranteed: swapping negates every feature, which negates `z`. A surface or round column, by contrast,
is the *same* whichever way round the players are, so on its own it can only learn coin-flip noise — the notebook shows it picking up
an "Olympics favour player 1" weight that is pure chance. Context like surface can still matter through **interactions**
("rank matters more in best-of-5") — a linear model needs those built by hand; trees (Phase 4) find them on their own.

**Missing values → 0.** `form_diff` is blank for a player's first match; 0 means "no difference", the neutral guess.

## The fitted model

| Feature | Weight | Odds × per 1 SD | 1 SD is about |
|---|---:|---:|---|
| `log_rank_diff` | −0.37 | 0.69 | 1.5 (e.g. #10 vs #45) |
| `surface_elo_diff` | +0.32 | 1.38 | 213 Elo |
| `elo_diff` | +0.26 | 1.29 | 212 Elo |
| `h2h_diff` | +0.07 | 1.07 | 1.5 wins |
| `rank_points_diff` | +0.05 | 1.05 | 2,455 points |
| `form_diff` | +0.04 | 1.04 | 0.24 (≈ 2.4 more wins in the last 10) |

- **Rank and Elo do the heavy lifting.** `log_rank_diff` is negative because positive means p1 is ranked *worse*.
- **Surface Elo outweighs overall Elo** — knowing how someone plays on *this* surface adds something.
- **Form and H2H barely register** once rank and Elo are known: most of what they say is already in the ratings.
- **Careful reading correlated features.** `rank_points_diff` and `log_rank_diff` are 83% correlated — they say almost the
  same thing. With weaker regularization (C = 1) `rank_points_diff` flips to **−0.12**, which would absurdly mean "more points
  hurts". The model just splits shared credit between twins arbitrarily. Read weights of correlated features as a group.

## Worked example — 2025 US Open final

Sinner (p1, #1) vs Alcaraz (p2, #2). Each feature, standardized, times its weight:

| Feature | Raw | Standardized | × Weight | = Contribution |
|---|---:|---:|---:|---:|
| `log_rank_diff` | −0.69 (#1 vs #2) | −0.47 | −0.37 | **+0.17** |
| `rank_points_diff` | +1,890 | +0.77 | +0.05 | +0.04 |
| `elo_diff` | +30 | +0.14 | +0.26 | +0.04 |
| `surface_elo_diff` | +162 (hard) | +0.76 | +0.32 | **+0.25** |
| `form_diff` | −0.10 | −0.42 | +0.04 | −0.02 |
| `h2h_diff` | −4 (Alcaraz led 9–5) | −2.68 | +0.07 | **−0.19** |
| **z** | | | | **+0.29** |

`p = 1 / (1 + e^−0.29)` = **57% Sinner**. His hard-court Elo and the #1 ranking pushed one way, Alcaraz's head-to-head lead the other.
Alcaraz won in four sets — a 43% outcome, the kind of result a well-calibrated model expects to see often.

## Results on validation (Apr–Dec 2025)

| Model | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| Naive (better rank wins) | 65.4% | 0.645 | 0.226 |
| Elo formula | 64.6% | 0.629 | 0.219 |
| **Logistic regression** | **65.4%** | **0.610** | **0.212** |

- **Probabilities: clear win.** Log-loss drops from 0.645 to 0.610. The naive rule can only ever say "66% for the favourite";
  logistic regression says 97% for a mismatch and 52% for a toss-up, and those numbers are well calibrated (notebook section 8).
- **Picking winners: a tie.** Exactly the same accuracy. The two disagree on 250 of 1,872 matches, and logistic regression is right on
  half of those — a coin flip. With 1,872 matches the margin of error on an accuracy gap is about ±1.7 points, so small differences
  aren't real either way.

That's typical in tennis: the favourite usually wins, and every sensible model picks the favourite. Better models mostly win on
**how confident** they are, not on **who** they pick — so compare log-loss first.
