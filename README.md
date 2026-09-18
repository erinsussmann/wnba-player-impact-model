# WNBA Player Impact

A RAPM-style (Regularized Adjusted Plus-Minus) player-impact model built from real
2025 WNBA play-by-play data, with a lineup-projection tool and an interactive
dashboard. Built as a portfolio project for a Data Scientist application.

**[Live dashboard](#)** <!-- TODO: replace with the deployed Render URL -->

## What this is

Raw plus-minus (how the team did while a player was on court) confounds a player's
own impact with their teammates' and opponents'. RAPM fixes this by regressing
point-differential on *who was on the floor*, across every possession in a season,
so a player's coefficient is their estimated impact holding the other 9 players on
the floor constant. It's the standard approach when you need lineup-level impact
rather than box-score stats, and it's the "lineup optimization" and "player impact"
core of what this application is scoped around.

## Methodology

1. **Data**: full 2025 WNBA season (312 games) pulled directly from
   [sportsdataverse's published parquet releases](https://github.com/sportsdataverse/sportsdataverse-data)
   — the same ESPN-backed play-by-play, schedules, and box scores that back the
   `wehoop`/`sportsdataverse` R and Python packages. See
   [Why not the `sportsdataverse` package](#why-not-the-sportsdataverse-package) below.

2. **Possession reconstruction** (`src/possessions.py`): the most defensible technical
   work here. Each team's on-court 5 is tracked by walking substitution events
   (seeded from box-score starters), and possessions are segmented on made shots,
   turnovers, and defensive rebounds (offensive rebounds continue the same
   possession). The tricky part is and-1s: a made basket immediately followed by a
   "Shooting Foul" event is *not* reliably an and-1 — a made basket followed by the
   *other* team's very next possession also drawing a shooting foul on their own
   drive looks byte-for-byte identical in the event stream. This was a real bug
   caught during validation (see below); it's resolved by checking whether the
   ensuing free-throw shooter is actually on the team that just scored.

3. **RAPM model** (`src/rapm_model.py`): one-hot encode the 10 on-court players per
   possession (+1 offense, -1 defense), target = points scored, fit with
   `sklearn.linear_model.Ridge`. Regularization strength is tuned via 5-fold
   cross-validation over a log-spaced grid, not guessed — collinearity between
   frequent lineup-mates and a short WNBA season (relative to the NBA's 82 games)
   both make regularization load-bearing here.

4. **Lineup tool** (`src/lineup_tool.py`): sums fitted coefficients for a 5-player
   lineup, and brute-forces the best 5-man combination from a given roster.

5. **Dashboard** (`dashboard/app.py`): a Dash app with a sortable, team-filterable
   ratings table, a top/bottom net-rating chart, an interactive lineup projector,
   and the validation scatter plot described next.

## Validation

Two independent checks, not just "the model ran without errors":

- **Possession reconstruction vs. box scores**: reconstructed points-per-team-per-game
  match the official box score exactly in 69.5% of team-games, and within ±5 points
  in 100% of them (n=558, across 279 games). Total combined points per game (both
  teams) match *exactly* in all 279 games — so no points are being lost, only
  occasionally misattributed between the two teams in edge cases (see
  Limitations). 33 of 312 games were excluded entirely after an automated check
  caught source-data substitution inconsistencies (a "player enters" event for
  someone already marked on-court, or a "player exits" event for someone who wasn't)
  — rather than risk training on a corrupted lineup, those games are dropped and the
  count is logged.

- **Model output vs. published RAPM**: this model's ratings are compared against the
  independently-computed `wnba_player_impact` dataset that `sportsdataverse`
  publishes (its own possession-engine RAPM/SPM/BPM/DARKO ratings). Since that
  dataset uses WNBA Stats player IDs and this project's data uses ESPN athlete IDs,
  the comparison goes through a published ESPN↔WNBA-Stats player crosswalk.
  Result: **Pearson r = 0.66, Spearman rank r = 0.62** across 213 matched players —
  meaningful agreement (this isn't noise), but not suspiciously close to 1.0, which
  would actually be a red flag given the two models use different possession
  definitions, regularization, and training data. A'ja Wilson and Breanna Stewart
  — the correct top two players by most public metrics that season — top this
  model's ratings too.

## Limitations

- **Single combined net-rating model**, not split offensive/defensive RAPM. The
  possession data supports fitting separate O and D models (regress points-scored
  on offense-only dummies, points-allowed on defense-only dummies); that's the
  natural next step and would let the lineup tool optimize for something like
  "best defensive 5" specifically.
- **And-1 / foul edge cases**: technical free throws are scored but never open or
  close a possession; flagrant-foul possession retention (the fouled team keeps
  the ball afterward) isn't modeled. These account for the residual ±5 point
  per-game-team discrepancy noted above.
- **One season, one league**: 43K possessions is enough for regularized ratings but
  small relative to an NBA-scale RAPM. Multi-season data would substantially
  tighten the fit, especially for low-minute players.
- **33 games excluded** for source-data substitution inconsistencies — see
  Validation above. This is a data-quality issue in the upstream feed, not a bug in
  this project's reconstruction (verified by inspecting the specific substitution
  events that failed the on/off-court consistency check).

## What I'd do with more time

- Split offensive/defensive RAPM.
- Multi-season data (the loaders already support any season — see `src/data_pull.py`).
- Incorporate shot-location data (already pulled in `pbp`'s `coordinate_x/y` fields)
  for a shot-quality-adjusted variant.
- Try to recover the 33 excluded games by repairing the specific bad substitution
  event rather than dropping the whole game.

## Why not the `sportsdataverse` package

The plan going in was to `pip install sportsdataverse` and use its WNBA loaders
directly. In practice, importing it pulls in every other sport it supports,
including college football's win-probability model, which imports `xgboost` —
and `xgboost`'s compiled binary needs a system OpenMP runtime (`brew install
libomp` on macOS) this machine didn't have. Separately, one of its NBA modules
uses `typing.TypeAlias`, a Python 3.10+ feature, and this project's only available
Python was 3.9. Rather than install Homebrew or a second Python just to satisfy an
unrelated dependency chain, `src/data_pull.py` fetches the same published parquet
files `sportsdataverse` would have, directly, with plain `pandas.read_parquet(url)`.

## Project layout

```
data/          raw and processed data (gitignored)
src/           pipeline code
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
