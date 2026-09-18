"""Fit a RAPM-style (Regularized Adjusted Plus-Minus) player-impact model.

For each possession, one-hot encode the 10 on-court players: +1 for each of
the 5 offensive players, -1 for each of the 5 defensive players. Target is
points scored on that possession. A Ridge regression's coefficients are then
each player's estimated net point impact per possession, holding teammates
and opponents constant -- the "adjusted" part of adjusted plus-minus, and
the reason it's preferred over raw on/off splits (lineups are highly
collinear; some players never play without each other).

This fits a single combined net-rating model (the MVP the project plan
calls for), not separate offensive/defensive RAPM models -- see README
limitations for why, and what a split model would add.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def build_feature_matrix(possessions: pd.DataFrame):
    players = sorted(
        set().union(*possessions["offense_lineup"]) | set().union(*possessions["defense_lineup"])
    )
    player_index = {p: i for i, p in enumerate(players)}

    n = len(possessions)
    rows, cols, vals = [], [], []
    for i, (off, dfn) in enumerate(zip(possessions["offense_lineup"], possessions["defense_lineup"])):
        for p in off:
            rows.append(i)
            cols.append(player_index[p])
            vals.append(1.0)
        for p in dfn:
            rows.append(i)
            cols.append(player_index[p])
            vals.append(-1.0)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(players)))
    y = possessions["points"].to_numpy(dtype=float)
    return X, y, players


def fit_rapm(season: int, force: bool = False) -> pd.DataFrame:
    out_path = PROCESSED_DIR / f"rapm_ratings_{season}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    possessions = pd.read_parquet(PROCESSED_DIR / f"possessions_{season}.parquet")
    X, y, players = build_feature_matrix(possessions)

    alphas = np.logspace(1, 5, 25)  # possession-level RAPM needs heavy regularization
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        Ridge(), param_grid={"alpha": alphas}, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    search.fit(X, y)
    best_alpha = search.best_params_["alpha"]
    print(f"[cv] best alpha: {best_alpha:.2f} (mean CV MSE: {-search.best_score_:.4f})")

    model = Ridge(alpha=best_alpha)
    model.fit(X, y)

    off_poss = np.asarray((X > 0).sum(axis=0)).ravel()
    def_poss = np.asarray((X < 0).sum(axis=0)).ravel()

    player_box = pd.read_parquet(RAW_DIR / f"player_box_{season}.parquet")
    team_box = pd.read_parquet(RAW_DIR / f"team_box_{season}.parquet")
    # Exclude exhibition teams (e.g. All-Star Game draft teams) so a
    # player's displayed team isn't overwritten by a one-off appearance --
    # see the matching filter and comment in possessions.build_possessions.
    games_played = team_box.groupby("team_id")["game_id"].nunique()
    real_teams = set(games_played[games_played >= 10].index)
    player_box = player_box[player_box["team_id"].isin(real_teams)]
    name_map = (
        player_box.sort_values("game_id")
        .groupby("athlete_id")
        .agg(player_name=("athlete_display_name", "last"), team_id=("team_id", "last"))
    )

    ratings = pd.DataFrame(
        {
            "athlete_id": players,
            "rapm": model.coef_,
            "off_possessions": off_poss,
            "def_possessions": def_poss,
        }
    ).set_index("athlete_id").join(name_map).reset_index()

    ratings = ratings.sort_values("rapm", ascending=False).reset_index(drop=True)
    ratings.attrs["best_alpha"] = best_alpha
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(out_path)
    print(f"[saved] {len(ratings)} player ratings -> {out_path}")
    return ratings


def load_published_ratings(season: int) -> pd.DataFrame:
    url = (
        "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        f"wnba_player_impact/wnba_player_impact_{season}.parquet"
    )
    cache = RAW_DIR / f"published_player_impact_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_parquet(url)
    df.to_parquet(cache)
    return df


def load_player_crosswalk() -> pd.DataFrame:
    """ESPN athlete_id <-> WNBA Stats player_id, needed because our
    possession data (ESPN) and the published RAPM validation data (WNBA
    Stats) use two entirely different player ID namespaces. This is a
    cached "current roster" snapshot (only the latest season is published,
    not one per historical season) but ESPN/WNBA-Stats IDs are stable
    across a player's career, so it still maps most players correctly.
    """
    url = (
        "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        "wnba_crosswalk/wnba_player_crosswalk_2026.parquet"
    )
    cache = RAW_DIR / "wnba_player_crosswalk.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_parquet(url)
    df.to_parquet(cache)
    return df


def compare_to_published(season: int, ratings: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    out_path = PROCESSED_DIR / f"rapm_validation_{season}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    published = load_published_ratings(season)
    crosswalk = load_player_crosswalk()[["espn_athlete_id", "wnba_player_id"]].dropna()
    crosswalk["espn_athlete_id"] = crosswalk["espn_athlete_id"].astype(float)
    crosswalk["wnba_player_id"] = crosswalk["wnba_player_id"].astype(int)

    ratings_xw = ratings.merge(crosswalk, left_on="athlete_id", right_on="espn_athlete_id")
    merged = ratings_xw.merge(
        published[["player_id", "rapm", "o_rapm", "d_rapm"]],
        left_on="wnba_player_id",
        right_on="player_id",
        suffixes=("_mine", "_published"),
    )
    merged["rank_mine"] = merged["rapm_mine"].rank(ascending=False)
    merged["rank_published"] = merged["rapm_published"].rank(ascending=False)
    corr = merged[["rapm_mine", "rapm_published"]].corr().iloc[0, 1]
    rank_corr = merged[["rank_mine", "rank_published"]].corr(method="spearman").iloc[0, 1]
    print(f"[validation] n={len(merged)} players matched to published ratings")
    print(f"[validation] Pearson corr (rapm): {corr:.3f}")
    print(f"[validation] Spearman rank corr: {rank_corr:.3f}")

    merged.to_parquet(out_path)
    print(f"[saved] validation comparison table -> {out_path}")
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ratings = fit_rapm(args.season, force=args.force)
    print(ratings.head(15).to_string(index=False))
    compare_to_published(args.season, ratings)
