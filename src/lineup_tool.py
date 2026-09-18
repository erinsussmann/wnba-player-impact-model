"""Project a 5-player lineup's net rating, and search a roster for the best 5.

Net rating here is the sum of the 5 players' fitted RAPM coefficients,
expressed per 100 possessions (RAPM itself is fit in points-per-possession
units; x100 just makes the numbers readable, matching the usual
"plus-minus per 100 possessions" convention).
"""
from itertools import combinations

import pandas as pd

from .rapm_model import fit_rapm

PER_100 = 100


def project_lineup(ratings: pd.DataFrame, player_names: list[str]) -> float:
    """Sum fitted RAPM coefficients for the given players (matched by name).

    Raises ValueError listing any name that doesn't match a rated player.
    """
    lookup = ratings.set_index("player_name")["rapm"]
    missing = [p for p in player_names if p not in lookup.index]
    if missing:
        raise ValueError(f"no rating for: {missing}")
    return float(lookup.loc[player_names].sum()) * PER_100


def best_lineup(ratings: pd.DataFrame, roster: list[str], size: int = 5) -> tuple[list[str], float]:
    """Brute-force the best-projected `size`-player lineup from a roster.

    Fine at roster scale (10-12 players -> a few hundred combinations);
    would need a smarter search for anything larger.
    """
    lookup = ratings.set_index("player_name")["rapm"]
    missing = [p for p in roster if p not in lookup.index]
    if missing:
        raise ValueError(f"no rating for: {missing}")

    best_combo, best_score = None, float("-inf")
    for combo in combinations(roster, size):
        score = lookup.loc[list(combo)].sum() * PER_100
        if score > best_score:
            best_combo, best_score = combo, score
    return list(best_combo), best_score


def team_roster(season: int, team_id) -> list[str]:
    ratings = fit_rapm(season)
    return sorted(ratings.loc[ratings["team_id"] == team_id, "player_name"].dropna().tolist())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--team-id", type=int, required=True)
    args = parser.parse_args()

    ratings = fit_rapm(args.season)
    roster = team_roster(args.season, args.team_id)
    print(f"roster ({len(roster)}): {roster}")
    combo, score = best_lineup(ratings, roster)
    print(f"best 5: {combo}")
    print(f"projected net rating: {score:+.1f} per 100 possessions")
