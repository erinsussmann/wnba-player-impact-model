# WNBA Player Impact

A RAPM (Regularized Adjusted Plus-Minus) model for the 2025 WNBA season, built from
play-by-play data, plus a lineup projection tool and a dashboard.

**[Live dashboard](#)** <!-- TODO: replace with the deployed Render URL -->

## Overview

Raw plus-minus confounds a player's impact with their teammates' and opponents' —
it just measures how the team did while they were on court. RAPM separates this out
by regressing point differential on who's on the floor across every possession in a
season, so each player's coefficient is their estimated impact holding the other
nine players constant.

## Data

Full 2025 WNBA season: play-by-play, box scores, and schedules for all 312 games,
pulled from [sportsdataverse's published data releases](https://github.com/sportsdataverse/sportsdataverse-data)
and cached locally as parquet.

## Methodology

1. **Possession reconstruction** (`src/possessions.py`) — tracks each team's
   on-court five by walking substitution events, seeded from box-score starters.
   Possessions are segmented on made shots, turnovers, and defensive rebounds;
   offensive rebounds continue the same possession. And-1s need a bit of care: a
   made basket immediately followed by a "Shooting Foul" event isn't necessarily an
   and-1 — a made basket followed by the *other* team's next possession drawing a
   shooting foul on their own drive looks identical in the event stream. This is
   resolved by checking whether the ensuing free-throw shooter is on the team that
   just scored.

2. **RAPM model** (`src/rapm_model.py`) — one-hot encodes the 10 on-court players
   per possession (+1 offense, -1 defense), target is points scored, fit with
   `sklearn.linear_model.Ridge`. Alpha is tuned via 5-fold cross-validation over a
   log-spaced grid.

3. **Lineup tool** (`src/lineup_tool.py`) — sums fitted coefficients for a 5-player
   lineup, and can brute-force the best 5-man combination from a roster.

4. **Dashboard** (`dashboard/app.py`) — a Dash app with a filterable ratings table,
   a top/bottom net-rating chart, a lineup projector, and the validation chart below.

## Validation

- **Possession reconstruction vs. box scores**: reconstructed points per team per
  game match the official box score exactly in 69.5% of team-games, and within ±5
  points in all of them (n=558, 279 games). Combined game totals (both teams) match
  exactly in every one of those 279 games, so the discrepancies are misattribution
  between the two teams on edge cases, not lost points. 33 of 312 games were
  excluded after an automated check flagged substitution events that didn't match
  the current on-court lineup — a source-data issue, not something worth training
  on.

- **Model output vs. published RAPM**: compared against `sportsdataverse`'s own
  published RAPM/SPM/BPM/DARKO ratings, joined through an ESPN↔WNBA-Stats player ID
  crosswalk since the two datasets use different ID systems. Pearson r = 0.66,
  Spearman rank r = 0.62 across 213 matched players. A'ja Wilson and Breanna
  Stewart top both models.

## Limitations

- Single combined net-rating model, not split offense/defense RAPM. The possession
  data supports fitting those separately (points-scored on offense-only dummies,
  points-allowed on defense-only dummies), which would also let the lineup tool
  optimize for something like "best defensive five" specifically.
- Technical free throws are scored but never open or close a possession, and
  flagrant-foul possession retention isn't modeled. Accounts for the residual
  per-game point discrepancies above.
- One season, ~43K possessions — enough for regularized ratings but small relative
  to an NBA-scale RAPM, especially for low-minute players.
- 33 games excluded for substitution-tracking inconsistencies in the source data.

## With more time

- Split offensive/defensive RAPM.
- Multi-season data (the loaders already take any season).
- A shot-quality-adjusted variant using the shot-location data already in the pbp.
- Repair the specific bad substitution event in the excluded games instead of
  dropping them outright.

## Project layout

```
data/          raw and processed data (gitignored)
src/
  data_pull.py       fetch + cache schedule/pbp/box scores
  possessions.py     reconstruct lineups and possessions from play-by-play
  rapm_model.py      fit RAPM, validate against published ratings
  lineup_tool.py     project/search lineup net ratings
dashboard/
  app.py             Dash app
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_pull --season 2025
python -m src.possessions --season 2025
python -m src.rapm_model --season 2025
python dashboard/app.py   # http://localhost:8050
```
